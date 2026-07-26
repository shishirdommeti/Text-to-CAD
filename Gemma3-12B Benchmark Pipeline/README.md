# Gemma 3 12B CAD Generation Pipeline

This repository archives the automated benchmarking pipeline designed to evaluate the Gemma 3 12B model's ability to generate CadQuery Python scripts from a custom set of engineering prompts. 

## Pipeline Architecture

The benchmark processes a provided Excel file containing raw natural language CAD descriptions. Because the original data lacked specific identifiers, the pipeline has IDs for each prompt to track execution and logging. 

**Core Mechanism: The Auto-Correction Loop**
Unlike standard zero-shot benchmarks, this pipeline features an automated compiler feedback loop. The LLM processes the prompts directly—without chain-of-thought (CoT) reasoning—and generates CadQuery code. The code is then executed in an isolated Python sandbox. If a syntax or API error occurs, the traceback is captured and fed back to the LLM, allowing the model to iteratively debug and self-correct its output before finalizing the geometry.

---

## Repository Structure

*   `README.md` — This documentation.
*   `environment.yml` — Exact Python and CUDA dependencies.
*   `iterative_inference_strict.py` — The main script containing the synthetic ID generator, sandbox execution, and the LLM auto-correction loop, as well as the STL generation.
*   `EvaluationPrompts(Reasoning).xlsx` — The source dataset containing the text prompts.

---

## Reproduction Steps

1.  **Environment Setup**

Because CadQuery requires the OpenCASCADE kernel, this project relies on Conda for dependency management. 

To recreate the exact execution environment, run:

1. `conda env create -f environment.yml`
2. `conda activate gemma3-12b`

2.  **Execution**
    Run the main pipeline script. The script will ingest the Excel file, assign synthetic IDs, and begin the generation and auto-correction cycles.
    ```bash
    python iterative_inference_strict.py
    ```

## Evaluation Criteria
Success is measured by the model's ability to output executable code that produces a valid, non-empty 3D solid, either on the first attempt or after successfully debugging compiler errors via the iterative feedback loop.
