"""Patch the official ToG repo for local OpenAI-compatible inference.

This is the first step for a strict FLARE/ToG reproduction:
keep the official ToG search pipeline intact, but replace its legacy OpenAI
transport with a robust client that can talk to vLLM/Ollama-style endpoints.

Run this on the server after cloning ToG:

    python scripts/patch_tog_openai_compat.py \
      --tog_dir /root/autodl-tmp/paper_baselines/ToG

The patch is intentionally narrow:
  * only rewrites ``ToG/utils.py``'s ``run_llm`` function;
  * reads endpoint settings from environment variables;
  * leaves relation/entity pruning, ToG width/depth, and prompts unchanged.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


RUN_LLM_REPLACEMENT = r'''def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
    """OpenAI-compatible chat client for strict ToG reproduction.

    The original ToG code used the pre-1.0 OpenAI SDK and hard-coded a local
    llama branch. This version preserves the same function signature while
    allowing vLLM/Ollama/OpenAI-compatible servers via environment variables:

      TOG_OPENAI_API_BASE=http://127.0.0.1:8000/v1
      TOG_OPENAI_API_KEY=EMPTY
      TOG_OPENAI_MODEL=Qwen3.5-35B-A3B-FP8
      TOG_LLM_TIMEOUT_S=60
      TOG_LLM_RETRIES=3
    """
    import json
    import os
    import time
    import urllib.error
    import urllib.request

    api_base = os.environ.get("TOG_OPENAI_API_BASE", "http://127.0.0.1:8000/v1").rstrip("/")
    api_key = os.environ.get("TOG_OPENAI_API_KEY", opeani_api_keys or "EMPTY")
    model = os.environ.get("TOG_OPENAI_MODEL", engine)
    timeout_s = int(os.environ.get("TOG_LLM_TIMEOUT_S", "60"))
    retries = int(os.environ.get("TOG_LLM_RETRIES", "3"))

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an AI assistant that helps people find information."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    url = api_base + "/chat/completions"

    last_err = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8")
            obj = json.loads(body)
            return str(obj["choices"][0]["message"]["content"]).strip()
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                detail = ""
            last_err = f"HTTP {exc.code}: {detail[:300]}"
        except Exception as exc:
            last_err = str(exc)
        print(f"[ToG LLM] attempt {attempt}/{max(1, retries)} failed: {last_err}", flush=True)
        time.sleep(min(2 * attempt, 8))

    raise RuntimeError(f"ToG LLM call failed after {retries} retries: {last_err}")
'''


def patch_run_llm(utils_path: Path) -> bool:
    text = utils_path.read_text(encoding="utf-8")
    if "TOG_OPENAI_API_BASE" in text and "ToG LLM call failed" in text:
        return False

    pattern = re.compile(
        r"def run_llm\(prompt, temperature, max_tokens, opeani_api_keys, engine=\"gpt-3\.5-turbo\"\):.*?(?=def construct_relation_prune_prompt)",
        flags=re.DOTALL,
    )
    new_text, n = pattern.subn(RUN_LLM_REPLACEMENT + "\n", text)
    if n != 1:
        raise RuntimeError(
            f"Could not locate exactly one legacy run_llm block in {utils_path}; matched {n}."
        )
    utils_path.with_suffix(".py.bak_tog_openai_compat").write_text(text, encoding="utf-8")
    utils_path.write_text(new_text, encoding="utf-8")
    return True


def patch_entity_prompt_signature(utils_path: Path) -> bool:
    """Make the official helper tolerant to its historical extra score argument."""
    text = utils_path.read_text(encoding="utf-8")
    old = "def construct_entity_score_prompt(question, relation, entity_candidates):"
    new = "def construct_entity_score_prompt(question, relation, entity_candidates, *unused):"
    if old not in text:
        return False
    utils_path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def patch_official_pre_head_typo(utils_path: Path) -> bool:
    """Fix a syntax typo present in the historical ToG utils.py."""
    text = utils_path.read_text(encoding="utf-8")
    old = "if not pre head or rel not in pre_relations"
    new = "if not pre_head or rel not in pre_relations"
    if old not in text:
        return False
    utils_path.write_text(text.replace(old, new), encoding="utf-8")
    return True


def patch_optional_sentence_transformers(utils_path: Path) -> bool:
    """Make sentence-transformers optional for ``--prune_tools llm`` runs."""
    text = utils_path.read_text(encoding="utf-8")
    optional_block = """try:
    from sentence_transformers import util, SentenceTransformer
except Exception:
    util = None
    SentenceTransformer = None"""

    if optional_block in text:
        return False

    # Handles both the original two-line import and a partially patched broken block.
    pattern = re.compile(
        r"(?:try:\s*)?\n?from sentence_transformers import util\s*\n"
        r"(?:except Exception:\s*\n\s*util = None\s*\n)?"
        r"(?:try:\s*)?\n?from sentence_transformers import SentenceTransformer\s*\n"
        r"(?:except Exception:\s*\n\s*SentenceTransformer = None\s*\n)?",
        flags=re.MULTILINE,
    )
    new_text, n = pattern.subn(optional_block + "\n", text, count=1)
    if n == 0:
        return False
    utils_path.write_text(new_text, encoding="utf-8")
    return True


def patch_freebase_retry(freebase_path: Path) -> bool:
    """Add retry/backoff around Virtuoso/SPARQL calls.

    Official ToG assumes a stable private Freebase service. In practice, Virtuoso
    can return transient 409/5xx errors when it is busy, so strict reproduction
    runs need retry diagnostics rather than crashing on the first query.
    """
    text = freebase_path.read_text(encoding="utf-8")
    if "TOG_SPARQL_RETRIES" in text and "HTTP 409" in text:
        return False

    if "import os" not in text:
        text = text.replace("from SPARQLWrapper import SPARQLWrapper, JSON", "from SPARQLWrapper import SPARQLWrapper, JSON\nimport os\nimport time\nimport urllib.error", 1)

    execute_replacement = '''def execurte_sparql(sparql_txt):
    retries = int(os.environ.get("TOG_SPARQL_RETRIES", "5"))
    timeout_s = int(os.environ.get("TOG_SPARQL_TIMEOUT_S", "60"))
    last_err = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            sparql = SPARQLWrapper(SPARQLPATH)
            try:
                sparql.setTimeout(timeout_s)
            except Exception:
                pass
            sparql.setQuery(sparql_txt)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            return results["results"]["bindings"]
        except urllib.error.HTTPError as exc:
            last_err = f"HTTP {exc.code}: {exc.reason}"
            if exc.code in (409, 429, 500, 502, 503, 504) and attempt < retries:
                print(f"[ToG SPARQL] transient {last_err}; retry {attempt}/{retries}", flush=True)
                time.sleep(min(2 * attempt, 10))
                continue
            raise
        except Exception as exc:
            last_err = str(exc)
            if attempt < retries:
                print(f"[ToG SPARQL] query failed: {last_err}; retry {attempt}/{retries}", flush=True)
                time.sleep(min(2 * attempt, 10))
                continue
            raise RuntimeError(
                f"Freebase SPARQL failed after {retries} retries. "
                f"Check SPARQLPATH={SPARQLPATH!r} and whether Virtuoso/Freebase is running. "
                f"Last error: {last_err}"
            )
'''

    pattern = re.compile(r"def execurte_sparql\(sparql_txt\):.*?(?=def replace_relation_prefix)", flags=re.DOTALL)
    text2, n = pattern.subn(execute_replacement + "\n", text)
    if n != 1:
        raise RuntimeError(f"Could not patch execurte_sparql in {freebase_path}; matched {n}.")

    id2_replacement = '''def id2entity_name_or_type(entity_id):
    sparql_txt = sparql_id % (entity_id, entity_id)
    retries = int(os.environ.get("TOG_SPARQL_RETRIES", "5"))
    timeout_s = int(os.environ.get("TOG_SPARQL_TIMEOUT_S", "60"))
    last_err = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            sparql = SPARQLWrapper(SPARQLPATH)
            try:
                sparql.setTimeout(timeout_s)
            except Exception:
                pass
            sparql.setQuery(sparql_txt)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            if len(results["results"]["bindings"]) == 0:
                return "UnName_Entity"
            return results["results"]["bindings"][0]['tailEntity']['value']
        except urllib.error.HTTPError as exc:
            last_err = f"HTTP {exc.code}: {exc.reason}"
            if exc.code in (409, 429, 500, 502, 503, 504) and attempt < retries:
                print(f"[ToG SPARQL] transient {last_err}; retry {attempt}/{retries}", flush=True)
                time.sleep(min(2 * attempt, 10))
                continue
            raise
        except Exception as exc:
            last_err = str(exc)
            if attempt < retries:
                print(f"[ToG SPARQL] name lookup failed: {last_err}; retry {attempt}/{retries}", flush=True)
                time.sleep(min(2 * attempt, 10))
                continue
            raise RuntimeError(
                f"Freebase name lookup failed after {retries} retries. "
                f"Check SPARQLPATH={SPARQLPATH!r}. Last error: {last_err}"
            )
'''
    pattern2 = re.compile(r"def id2entity_name_or_type\(entity_id\):.*\Z", flags=re.DOTALL)
    text3, n2 = pattern2.subn(id2_replacement + "\n", text2)
    if n2 != 1:
        raise RuntimeError(f"Could not patch id2entity_name_or_type in {freebase_path}; matched {n2}.")

    freebase_path.with_suffix(".py.bak_tog_sparql_retry").write_text(text, encoding="utf-8")
    freebase_path.write_text(text3, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tog_dir", required=True, help="Path to the cloned official ToG repo.")
    args = parser.parse_args()

    tog_dir = Path(args.tog_dir).expanduser().resolve()
    utils_path = tog_dir / "ToG" / "utils.py"
    freebase_path = tog_dir / "ToG" / "freebase_func.py"
    main_path = tog_dir / "ToG" / "main_freebase.py"
    if not utils_path.exists():
        raise FileNotFoundError(f"Missing ToG utils.py: {utils_path}")
    if not main_path.exists():
        raise FileNotFoundError(f"Missing ToG main_freebase.py: {main_path}")
    if not freebase_path.exists():
        raise FileNotFoundError(f"Missing ToG freebase_func.py: {freebase_path}")

    changed_run_llm = patch_run_llm(utils_path)
    changed_sig = patch_entity_prompt_signature(utils_path)
    changed_pre_head = patch_official_pre_head_typo(utils_path)
    changed_st = patch_optional_sentence_transformers(utils_path)
    changed_sparql = patch_freebase_retry(freebase_path)
    print(
        {
            "tog_dir": str(tog_dir),
            "patched_run_llm": changed_run_llm,
            "patched_entity_prompt_signature": changed_sig,
            "patched_pre_head_typo": changed_pre_head,
            "patched_optional_sentence_transformers": changed_st,
            "patched_sparql_retry": changed_sparql,
            "utils_path": str(utils_path),
            "freebase_path": str(freebase_path),
        }
    )


if __name__ == "__main__":
    main()
