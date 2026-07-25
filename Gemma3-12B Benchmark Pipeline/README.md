# Gemma 3 12B CAD Generation Pipeline

This repository archives the automated benchmarking pipeline designed to evaluate the Gemma 3 12B model's ability to generate CadQuery Python scripts from a custom set of engineering prompts. 

## ⚙️ Pipeline Architecture

The benchmark processes a provided Excel file containing raw natural language CAD descriptions. Because the original data lacked specific identifiers, the pipeline automatically generates synthetic IDs for each prompt to track execution and logging. 

**Core Mechanism: The Auto-Correction Loop**
Unlike standard zero-shot benchmarks, this pipeline features an automated compiler feedback loop. The LLM processes the prompts directly—without chain-of-thought (CoT) reasoning—and generates CadQuery code. The code is then executed in an isolated Python sandbox. If a syntax or API error occurs, the traceback is captured and fed back to the LLM, allowing the model to iteratively debug and self-correct its output before finalizing the geometry.

---

## 📂 Repository Structure

*   `README.md` — This documentation.
*   `requirements.txt` — Exact Python and CUDA dependencies.
*   `benchmark_pipeline.py` — The main script containing the synthetic ID generator, sandbox execution, and the LLM auto-correction loop.
*   `advisor_prompts.xlsx` — The source dataset containing the text prompts.

*(Note: Do not commit the generated CAD models or any scripts containing hardcoded API tokens).*

---

## 🚀 Reproduction Steps

1.  **Environment Setup**
    Ensure a Linux environment with CUDA 12.1 and at least 24GB of VRAM for `bfloat16` inference.
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Execution**
    Run the main pipeline script. The script will ingest the Excel file, assign synthetic IDs, and begin the generation and auto-correction cycles.
    ```bash
    python benchmark_pipeline.py
    ```

## 📊 Evaluation Criteria
Success is measured by the model's ability to output executable code that produces a valid, non-empty 3D solid, either on the first attempt or after successfully debugging compiler errors via the iterative feedback loop.
