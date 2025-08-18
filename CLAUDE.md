# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Goedel-Prover-V2 is an open-source automated theorem prover for Lean 4 that achieves state-of-the-art performance in formal proof generation. The system uses Large Language Models (LLMs) with a self-correction mechanism to generate and verify mathematical proofs.

## Environment Setup Commands

### macOS Setup (ARM64)
```bash
# Create conda environment with Python 3.10
conda create -n goedelv2 python=3.10 -y
conda activate goedelv2

# Install core dependencies
pip install torch==2.6.0 torchvision transformers accelerate ipykernel jupyter_client

# Install flash-attn if needed (may fail on macOS, optional)
pip install flash-attn --no-build-isolation

# Build Mathlib4
cd mathlib4
lake build
cd ..

# Test Lean installation
python lean_compiler/repl_scheduler.py
```

### Linux Setup
```bash
# Use the original environment file (Linux-specific)
conda env create -f goedelv2.yml
conda activate goedelv2

# Build Mathlib4
cd mathlib4
lake build
cd ..

# Test installation
python lean_compiler/repl_scheduler.py
```

## Key Commands

### Running the Complete Pipeline
```bash
# Main pipeline for inference, compilation, and summarization
bash scripts/pipeline.sh

# Pipeline configuration is done within scripts/pipeline.sh:
# - MODEL_PATH: Path or HuggingFace ID for the model (e.g., "Goedel-LM/Goedel-Prover-V2-32B")
# - DATA_PATH: Input problems file (e.g., "dataset/minif2f.jsonl")
# - GPUS: Number of GPUs for inference
# - CPUS: Number of CPUs for compilation
# - MAX_CORRECTION_ROUNDS: Number of self-correction iterations
```

### Individual Components

```bash
# 1. Inference - Generate proofs using LLM
python src/inference.py \
    --model_path "Goedel-LM/Goedel-Prover-V2-32B" \
    --input_path dataset/test.jsonl \
    --output_dir results/test \
    --n 32 \
    --gpu 4 \
    --inference_handler dpskcot \
    --correction_round 0 \
    --max_model_len 40960 \
    --temp 1.0

# 2. Compilation - Verify generated proofs with Lean
python src/compile.py \
    --input_path results/test/to_inference_codes.json \
    --output_path results/test/code_compilation_repl.json \
    --cpu 32

# 3. Summarization - Generate performance reports
python src/summarize.py \
    --input_path results/test/code_compilation_repl.json \
    --full_record_path results/test/full_records.json \
    --output_dir results/test/summary
```

## Architecture Overview

### Core Pipeline Flow

1. **Inference Phase** (`src/inference.py`)
   - Uses vLLM for efficient LLM inference
   - Supports multiple inference handlers: `dpskcot`, `dpsknoncot`, `kiminacot`
   - Generates multiple proof candidates per problem (configurable via `--n`)
   - Handles self-correction rounds by feeding error information back to the model

2. **Compilation Phase** (`src/compile.py`)
   - Interfaces with Lean 4 through `lean_compiler/repl_scheduler.py`
   - Validates generated proofs using Lean's type checker
   - Runs in parallel using multiprocessing
   - Timeout protection (default 300s per proof, configurable via `PROOF_TIMEOUT` env var)

3. **Summarization Phase** (`src/summarize.py`)
   - Analyzes compilation results
   - Generates success metrics and problem-level statistics
   - Outputs CSV and JSON reports

### Self-Correction Mechanism

The system implements iterative proof refinement:
- **Round 0**: Initial proof generation from problem statements
- **Round 1+**: Corrections based on Lean compiler feedback
- Each correction round uses fewer samples (configurable)
- Maximum 2 rounds by default (40K tokens total)

### Key Configuration Points

- **Lean Workspace**: `mathlib4/` directory (must be built with `lake build`)
- **Default Lake Path**: `~/.elan/bin/lake`
- **Imports**: Standard Mathlib imports defined in `lean_compiler/repl_scheduler.py`
- **Timeout Settings**: 
  - Import timeout: 100s
  - Proof timeout: 300s (configurable via `PROOF_TIMEOUT`)

### Data Format

Input problems (JSONL format):
```json
{
  "name": "problem_id",
  "lean4_code": "theorem statement...",
  "origin_problem_id": "unique_identifier"
}
```

Output includes:
- `full_records.json`: Complete generation records
- `to_inference_codes.json`: Formatted proofs for compilation
- `code_compilation_repl.json`: Compilation results
- Summary reports in CSV and JSON formats

## Important Configuration Files

- `goedelv2.yml`: Conda environment specification (Linux-specific with build hashes)
- `lean_compiler/repl_scheduler.py`: Lean REPL interface configuration
- `scripts/pipeline.sh`: Main pipeline configuration and orchestration

## Model Information

- **Goedel-Prover-V2-32B**: Flagship model achieving 88.0% on MiniF2F (Pass@32)
- **Goedel-Prover-V2-8B**: Smaller model achieving 83.0% on MiniF2F (Pass@32)
- Models available on HuggingFace: `Goedel-LM/Goedel-Prover-V2-{32B,8B}`

## Troubleshooting

- **macOS conda environment issues**: Remove build hashes from `goedelv2.yml` or create fresh environment
- **Lean installation errors**: Ensure Lean 4 is installed and `lake` is in PATH
- **Mathlib build failures**: Check Lean version compatibility (requires Lean 4 version 4.9)
- **Memory issues during inference**: Reduce `--max_model_len` or batch size
- **Timeout issues**: Adjust `PROOF_TIMEOUT` environment variable

## Datasets

- `dataset/minif2f.jsonl`: MiniF2F benchmark problems
- `dataset/MOBench.jsonl`: Math Olympiad Bench (360 IMO-level problems)
- `dataset/test.jsonl`: Test dataset for quick validation