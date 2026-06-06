import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List


def _set_if_present(cfg: Dict[str, Any], keys: List[str], value: Any) -> bool:
    cur = cfg
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            return False
        cur = nxt
    if keys[-1] in cur:
        cur[keys[-1]] = value
        return True
    return False


def _patch_data_root(cfg: Dict[str, Any], data_root: str) -> None:
    # ALFWorld configs vary slightly across versions. Patch the common fields
    # when present and keep the default config otherwise.
    candidates = [
        ["dataset", "data_path"],
        ["dataset", "data_dir"],
        ["env", "data_path"],
        ["env", "data_dir"],
    ]
    root = Path(data_root)
    dataset_root = root / "json_2.1.1" if (root / "json_2.1.1").exists() else root
    for keys in candidates:
        _set_if_present(cfg, keys, str(dataset_root))


def _expand_env_vars(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def _as_batch(value: Any, batch_size: int) -> List[Any]:
    if isinstance(value, list):
        return value
    return [value for _ in range(batch_size)]


def _choose_action(commands: List[str], rng: random.Random) -> str:
    if not commands:
        return "look"
    preferred = [
        cmd
        for cmd in commands
        if cmd
        and cmd.lower() not in {"look", "inventory"}
        and not cmd.lower().startswith("examine")
    ]
    return rng.choice(preferred or commands)


def _find_config(data_root: Path, package_root: Path) -> Path | None:
    candidate_roots = [data_root, package_root, package_root.parent]
    names = ["base_config.yaml", "config.yaml", "alfworld_config.yaml"]
    for root in candidate_roots:
        for name in names:
            path = root / name
            if path.exists():
                return path
    for root in candidate_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.yaml"):
            if "config" in path.name.lower():
                return path
    return None


def _load_alfworld_config(generic: Any, config_path: Path) -> Dict[str, Any]:
    # ALFWorld's generic.load_config reads sys.argv internally. Temporarily
    # isolate it from this script's flags such as --data_root.
    old_argv = sys.argv[:]
    sys.argv = [old_argv[0], str(config_path)]
    try:
        return generic.load_config()
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny ALFWorld text-environment smoke test.")
    parser.add_argument("--data_root", type=str, default="data/long_horizon/alfworld")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--split", type=str, default="eval_out_of_distribution")
    parser.add_argument("--env_type", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    os.environ["ALFWORLD_DATA"] = str(data_root)
    rng = random.Random(int(args.seed))

    print(f"[alfworld] ALFWORLD_DATA={data_root}")
    print(f"[alfworld] data_exists={data_root.exists()}")
    if data_root.exists():
        print("[alfworld] top_level=", sorted(p.name for p in data_root.iterdir())[:20])

    import alfworld.agents.environment as environment
    import alfworld.agents.modules.generic as generic
    import alfworld

    package_root = Path(alfworld.__file__).resolve().parent
    config_path = Path(args.config).resolve() if args.config else _find_config(data_root, package_root)
    if not config_path or not config_path.exists():
        raise FileNotFoundError(
            "Could not find an ALFWorld YAML config. Try running:\n"
            "  grep -n \"URL\\|base_config\\|CONFIG\" $(which alfworld-download)\n"
            "or provide --config /path/to/base_config.yaml"
        )
    print(f"[alfworld] config={config_path}")
    cfg = _expand_env_vars(_load_alfworld_config(generic, config_path))
    _patch_data_root(cfg, str(data_root))
    env_type = args.env_type or cfg.get("env", {}).get("type", "AlfredTWEnv")
    print(f"[alfworld] env_type={env_type} split={args.split}")

    env_cls = environment.get_environment(env_type)
    env = env_cls(cfg, train_eval=args.split)
    env = env.init_env(batch_size=int(args.batch_size))

    obs, infos = env.reset()
    obs_batch = _as_batch(obs, int(args.batch_size))
    print("[alfworld] reset ok")
    print("[alfworld] initial_obs=", str(obs_batch[0])[:500].replace("\n", " "))

    for step in range(int(args.steps)):
        admissible = infos.get("admissible_commands", [[] for _ in range(int(args.batch_size))])
        actions = [_choose_action(list(cmds), rng) for cmds in admissible]
        obs, scores, dones, infos = env.step(actions)
        obs_batch = _as_batch(obs, int(args.batch_size))
        print(f"[alfworld] step={step + 1} action={actions[0]!r} score={scores[0]} done={dones[0]}")
        print("[alfworld] obs=", str(obs_batch[0])[:300].replace("\n", " "))
        if all(bool(x) for x in dones):
            break

    print("[alfworld] smoke test finished")


if __name__ == "__main__":
    main()
