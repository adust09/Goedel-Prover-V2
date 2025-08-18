# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Core Python modules — `inference.py` (LLM generation), `compile.py` (Lean verification), `summarize.py` (reports), `utils.py`.
- `lean_compiler/`: Lean REPL scheduler used by `compile.py`.
- `mathlib4/`: Lean mathlib submodule (build with `lake`).
- `scripts/`: Automation, e.g., `pipeline.sh` for batch inference→compile→summary.
- `dataset/`: Sample/problem inputs. `assets/`: figures. `results/`: outputs (git-ignored).

## Build, Test, and Development Commands
- Create env: `conda env create -f goedelv2.yml && conda activate goedelv2`.
- Build mathlib: `cd mathlib4 && lake build` (Lean toolchain required).
- Verify Lean REPL: `python lean_compiler/repl_scheduler.py`.
- Quick local demo: `python quick_start_demo.py`.
- Batch pipeline: `bash scripts/pipeline.sh` (edits go in the CONFIGURATION section).
- Individual steps:
  - Inference: `python src/inference.py --model_path Goedel-LM/Goedel-Prover-V2-8B --input_path dataset/test.jsonl --output_dir results/run_XXXX`
  - Compile: `python src/compile.py --input_path results/run_XXXX/to_inference_codes.json --output_path results/run_XXXX/code_compilation_repl.json --cpu 32`
  - Summarize: `python src/summarize.py --input_path results/run_XXXX/code_compilation_repl.json --full_record_path results/run_XXXX/full_records.json --output_dir results/run_XXXX/summary_round_0`

## Coding Style & Naming Conventions
- Language: Python 3.10; Lean 4 for proofs.
- Indentation: 4 spaces; keep lines < 100 chars where practical.
- Naming: `snake_case` for functions/vars, `CapWords` for classes, modules lowercase.
- I/O: prefer explicit args/paths; avoid hidden state. Add docstrings for new public functions.
- Formatting/lint: no enforced tool in repo; if used locally, Black/ruff are fine — do not reformat unrelated files.

## Testing Guidelines
- No formal unit test suite; validate via end-to-end runs:
  - API smoke test: `python test_api.py` (requires `HF_TOKEN`).
  - Pipeline accuracy: run `scripts/pipeline.sh` and inspect `summary_round_*/meta_summarize.json` and `*_summarize.csv`.
- Test data: place JSON/JSONL under `dataset/`; keep outputs under `results/` (already git-ignored).

## Commit & Pull Request Guidelines
- Commits: concise, imperative mood (e.g., "Add compile timeout flag"). Group related changes.
- PRs: include purpose, key changes, how to run, and sample outputs/metrics (paths under `results/`). Link issues when applicable. Add screenshots or CSV snippets for summaries.
- Scope: avoid drive-by refactors; keep diffs minimal and focused.

## Security & Configuration Tips
- Do not commit secrets. Keep tokens in `.env` or environment (see `guide_api.md`); `.env` is git-ignored.
- Large outputs and models must not be committed. Use `results/` for artifacts and share pointers instead.
