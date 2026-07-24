# CADFusion Baseline Evaluation & Benchmarking Pipeline

This folder contains the benchmarking scripts, evaluation data, and execution pipeline used to evaluate the zero-shot spatial reasoning capabilities of **Microsoft's CADFusion** model (LLaMA-3-8B LoRA). 

The goal of this evaluation is to quantify the text-to-CAD generation success rate when the model's outputs are strictly validated through a physical geometry kernel (OpenCASCADE), rather than relying on surface-level text or syntax parsing.

---

## Project Overview & Methodological Caveats

This benchmark evaluates CADFusion's ability to generate valid SkexGen parametric command sequences under two strict conditions: **Direct Zero-Shot** and **Chain-of-Thought (CoT)** prompting.

---

## Prerequisites & Environment Setup

To recreate this benchmarking environment, you will need a GPU-enabled Python environment with access to the Hugging Face Transformers library and Microsoft's official CADFusion evaluation tools.

### 1. Python Dependencies
Install the required local inference libraries:
```bash
pip install torch transformers peft pandas openpyxl
```

### 2. Microsoft CADFusion Geometry Tools
To actually compile the output tokens into 3D `.step` files, you must download the official parsing scripts from Microsoft's repository.
*   **Repository:** [Microsoft CADFusion GitHub](https://github.com/microsoft/CADFusion)
*   **Setup:** Clone their repository and follow their instructions to install the `OpenCASCADE` python dependencies. You will specifically need their visualizer/parser script (often named `generate_samples.sh` or the respective python parser) to convert the JSON token sequences into physical geometries.

---

## Execution Pipeline

The benchmarking process is divided into four distinct automated steps. Run the scripts in the following order:

### Step 1: LLM Inference Generation
Executes the model locally, resizing the token embeddings to accommodate the custom SkexGen vocabulary (128258), and generates the raw sequence tokens.
```bash
python run_cadfusion_benchmark.py
```
*   **Input:** `EvaluationPrompts(Reasoning).xlsx`
*   **Output:** `CADFusion_Raw_Outputs.xlsx`

### Step 2: Payload Extraction & JSON Packaging
Cleans the generated text of specialized system tokens (e.g., `<|eot_id|>`, `<eos>`) and formats the output into the exact JSON dictionary required by the Microsoft parser.
```bash
python export_to_json.py
```
*   **Input:** `CADFusion_Raw_Outputs.xlsx`
*   **Output:** `cad_outputs_v2.json`

### Step 3: Geometry Compilation (External)
This step uses Microsoft's OpenCASCADE tools. Feed the generated JSON file into their parser. The parser will attempt to execute the math; if the script is topologically sound (watertight, closed loops, no self-intersections), it will output a 3D file into a sequentially numbered directory (e.g., `visual_objects_v2/000000/`).
```bash
# Execute using Microsoft's provided evaluation scripts
python <microsoft_parser_script>.py --input cad_outputs_v2.json --output_dir visual_objects_v2
```

### Step 4: Empirical Result Mapping
Scans the generated 3D object folders. If a folder contains successfully compiled CAD files, the script maps that success back to the corresponding row in the original Excel sheet, creating the final graded master document.
```bash
python match_results_reasoning.py
```
*   **Input:** `visual_objects_v2/` & `CADFusion_Raw_Outputs.xlsx`
*   **Output:** `FINAL_Mapped_Reasoning_Results.xlsx`

---

## Baseline Findings

On a 57-prompt evaluation set, the CADFusion model demonstrated the following baseline performance when parsed through the strict geometry kernel:

| Prompting Strategy | Target Output Syntax | Total Prompts | Valid Compilations | Success Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Direct Zero-Shot** | SkexGen Tokens | 77 | 15 | **19.5%** |
| **Chain-of-Thought** | SkexGen Tokens | 77 | 18 | **23.4%** |

*Note: Failures were predominantly caused by "coordinate drift," where early mathematical miscalculations led to unclosed loops or impossible constraints that the geometry compiler rejected.*
