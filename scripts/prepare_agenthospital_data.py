import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.clinical_data import prepare_agenthospital_like, prepare_ai_hospital_repo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agenthospital",
        type=str,
        default="",
        help="Path to AgentHospital-like raw json/jsonl file.",
    )
    parser.add_argument(
        "--clinicalbench",
        type=str,
        default="",
        help="Optional second source path in AgentHospital-like format.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/clinical_processed")
    parser.add_argument(
        "--ai_hospital_root",
        type=str,
        default="",
        help="Path to cloned AI_Hospital repository root.",
    )
    parser.add_argument(
        "--ai_hospital_dialog_file",
        type=str,
        default="dialog_history_gpt4.jsonl",
        help="Dialog history filename under src/outputs/dialog_history_iiyi/.",
    )
    parser.add_argument(
        "--question_keys",
        type=str,
        default="instruction,question,prompt,task,chief_complaint,case_summary,text",
        help="Comma-separated fallback keys for question text.",
    )
    parser.add_argument(
        "--path_keys",
        type=str,
        default="oracle_path,gold_actions,action_path,plan,trajectory,steps",
        help="Comma-separated fallback keys for oracle action path list.",
    )
    parser.add_argument(
        "--step_stage_key",
        type=str,
        default="stage",
        help="Field name for stage in each step dict.",
    )
    parser.add_argument(
        "--step_action_keys",
        type=str,
        default="action,name,tool,intent,op",
        help="Comma-separated fallback keys for action token in each step dict.",
    )
    args = parser.parse_args()

    if args.ai_hospital_root:
        manifest = prepare_ai_hospital_repo(
            repo_root=args.ai_hospital_root,
            out_dir=args.out,
            seed=args.seed,
            dialog_file=args.ai_hospital_dialog_file,
        )
        print(f"Prepared AI_Hospital planning data at {args.out}")
        print(manifest)
        return

    sources = {}
    if args.agenthospital:
        sources["agenthospital"] = args.agenthospital
    if args.clinicalbench:
        sources["clinicalbench"] = args.clinicalbench

    if not sources:
        raise ValueError("Please provide at least one source: --agenthospital and/or --clinicalbench")

    field_mapping = {
        "question_keys": [x.strip() for x in str(args.question_keys).split(",") if x.strip()],
        "path_keys": [x.strip() for x in str(args.path_keys).split(",") if x.strip()],
        "step_stage_key": str(args.step_stage_key).strip(),
        "step_action_keys": [x.strip() for x in str(args.step_action_keys).split(",") if x.strip()],
    }

    manifest = prepare_agenthospital_like(
        data_sources=sources,
        out_dir=args.out,
        seed=args.seed,
        field_mapping=field_mapping,
    )
    print(f"Prepared clinical planning data at {args.out}")
    print(manifest)


if __name__ == "__main__":
    main()
