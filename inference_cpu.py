import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Make src/ importable when running from repo root
REPO_ROOT = Path(__file__).parent.resolve()
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils import DeepSeekCoTHandler, DeepSeekNonCoTHandler, KiminaCoTHandler  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="CPU-only inference using Hugging Face Transformers."
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Model ID or local path (e.g., Goedel-LM/Goedel-Prover-V2-8B)")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input JSONL with problems")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save outputs")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of samples per problem")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Max new tokens to generate per sample")
    parser.add_argument("--temp", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--inference_handler", type=str, default="dpskcot",
                        choices=["dpskcot", "dpsknoncot", "kiminacot"],
                        help="Prompting style / handler")
    parser.add_argument("--use_cpu", action="store_true",
                        help="Force CPU (default). Present for notebook parity.")
    return parser.parse_args()


def load_input_jsonl(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Set seed for reproducibility
    seed = random.randint(1, 99999)
    torch.manual_seed(seed)

    # Load tokenizer and model on CPU
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},  # ensure CPU
    )
    device = torch.device("cpu")

    # Select handler
    if args.inference_handler == "dpskcot":
        handler = DeepSeekCoTHandler()
    elif args.inference_handler == "dpsknoncot":
        handler = DeepSeekNonCoTHandler()
    elif args.inference_handler == "kiminacot":
        handler = KiminaCoTHandler()
    else:
        raise ValueError("Unsupported handler")

    # Load inputs
    data_list = load_input_jsonl(args.input_path)
    if not data_list:
        print("No input data found; exiting.")
        return

    all_records = []  # rich records for summarize
    to_inference = []  # minimal fields for compile

    for item in data_list:
        origin_id = item.get("origin_problem_id", item.get("problem_id", item.get("name", "problem")))
        lean4_code = item.get("lean4_code")
        if not lean4_code:
            # skip items without Lean code to prove
            continue

        # Build prompt/messages
        prompt_str, messages = handler.prover_inference(lean4_code, tokenizer)

        # Prepare inputs tensor
        if messages is not None:
            model_inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(device)
        else:
            model_inputs = tokenizer(prompt_str, return_tensors="pt").to(device)

        # Generate n samples
        for j in range(args.n):
            gen_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=args.max_length,
                do_sample=True,
                temperature=args.temp,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
            decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=False)[0]

            # Extract code and finalize full_code
            full_code = handler.extrac_code(decoded) if hasattr(handler, "extrac_code") else None
            if not full_code or full_code == "None":
                full_code = "None"
            else:
                full_code = handler.problem_check(lean4_code, full_code)

            gen_problem_id = f"{origin_id}_g{j}"
            record = {
                "origin_problem_id": origin_id,
                "problem_id": gen_problem_id,
                "id_maps": [
                    {"origin_problem_id": origin_id},
                    {"generation_id": gen_problem_id},
                ],
                "lean4_code": lean4_code,
                "model_input": prompt_str,
                "messages_history_list": messages if messages is not None else [],
                "model_output": decoded,
                "full_code": full_code,
            }
            all_records.append(record)
            to_inference.append({
                "problem_id": gen_problem_id,
                "origin_problem_id": origin_id,
                "id_maps": record["id_maps"],
                "lean4_code": lean4_code,
                "model_input": prompt_str,
                "messages_history_list": record["messages_history_list"],
                "model_output": decoded,
                "full_code": full_code,
            })

    # Save outputs consistent with compile/summarize expectations
    out_records = Path(args.output_dir) / "full_records.json"
    out_infer = Path(args.output_dir) / "to_inference_codes.json"
    with open(out_records, "w") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    with open(out_infer, "w") as f:
        json.dump(to_inference, f, indent=2, ensure_ascii=False)

    print("Outputs saved:")
    print(f"  Records: {out_records}")
    print(f"  Inference Codes: {out_infer}")


if __name__ == "__main__":
    main()

