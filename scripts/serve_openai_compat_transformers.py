#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat server backed by Hugging Face transformers.

This is intended as a robust fallback when vLLM is unavailable or incompatible
with the local GPU / CUDA stack. It implements only the endpoints needed by the
DiPLaN / ToG pipeline:

  - GET  /v1/models
  - POST /v1/chat/completions

Example:
  python scripts/serve_openai_compat_transformers.py \
    --model-path /root/autodl-tmp/Qwen2.5-7B-Instruct \
    --served-model-name Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _parse_dtype(name: str):
    import torch

    key = (name or "auto").lower()
    if key == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


@dataclass
class ServerState:
    model_path: str
    served_model_name: str
    device: str
    dtype_name: str
    max_input_tokens: int

    tokenizer: Any
    model: Any
    generation_lock: threading.Lock


def _load_state(args: argparse.Namespace) -> ServerState:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _parse_dtype(args.dtype)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    if device == "cpu":
        model.to("cpu")
    model.eval()

    return ServerState(
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        device=device,
        dtype_name=args.dtype,
        max_input_tokens=args.max_input_tokens,
        tokenizer=tokenizer,
        model=model,
        generation_lock=threading.Lock(),
    )


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _format_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            content = "\n".join(parts)
        out.append({"role": role, "content": str(content)})
    return out


def _build_prompt(state: ServerState, messages: list[dict[str, Any]]) -> str:
    formatted = _format_messages(messages)
    tokenizer = state.tokenizer
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            formatted,
            tokenize=False,
            add_generation_prompt=True,
        )

    lines: list[str] = []
    for msg in formatted:
        lines.append(f"{msg['role'].upper()}: {msg['content']}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def _generate(state: ServerState, body: dict[str, Any]) -> dict[str, Any]:
    import torch

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    prompt = _build_prompt(state, messages)
    tokenizer = state.tokenizer
    model = state.model

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=state.max_input_tokens)
    if state.device == "cuda":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    temperature = float(body.get("temperature", 0.0))
    max_tokens = int(body.get("max_tokens", 128))
    top_p = float(body.get("top_p", 1.0))
    do_sample = temperature > 0.0

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with state.generation_lock:
        with torch.no_grad():
            output = model.generate(**inputs, **generation_kwargs)

    prompt_len = int(inputs["input_ids"].shape[1])
    generated_ids = output[0][prompt_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": created,
        "model": str(body.get("model") or state.served_model_name),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": int(generated_ids.shape[0]),
            "total_tokens": prompt_len + int(generated_ids.shape[0]),
        },
    }


class OpenAICompatHandler(BaseHTTPRequestHandler):
    server_version = "DiPLaNTransformers/0.1"

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        raw = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        state: ServerState = self.server.state  # type: ignore[attr-defined]
        if self.path in {"/health", "/healthz"}:
            self._send_json({"status": "ok", "model": state.served_model_name})
            return
        if self.path == "/v1/models":
            self._send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": state.served_model_name,
                            "object": "model",
                            "created": 0,
                            "owned_by": "local-transformers",
                        }
                    ],
                }
            )
            return
        self._send_json({"error": {"message": f"unknown GET path: {self.path}"}}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._send_json({"error": {"message": f"unknown POST path: {self.path}"}}, status=HTTPStatus.NOT_FOUND)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": {"message": f"invalid JSON: {exc}"}}, status=HTTPStatus.BAD_REQUEST)
            return

        if bool(body.get("stream", False)):
            self._send_json(
                {"error": {"message": "stream=true is not supported by this fallback server"}},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        try:
            state: ServerState = self.server.state  # type: ignore[attr-defined]
            payload = _generate(state, body)
        except Exception as exc:  # noqa: BLE001
            print("[server] chat/completions failed:")
            print(traceback.format_exc(), flush=True)
            self._send_json({"error": {"message": str(exc)}}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json(payload)

    def log_message(self, fmt: str, *args: Any) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {self.address_string()} {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--served-model-name", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    state = _load_state(args)

    httpd = ThreadingHTTPServer((args.host, args.port), OpenAICompatHandler)
    httpd.state = state  # type: ignore[attr-defined]
    print(
        f"[server] listening on http://{args.host}:{args.port} "
        f"model={state.served_model_name} device={state.device} dtype={state.dtype_name}"
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()
