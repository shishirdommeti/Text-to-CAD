import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
import sys
import torch
import subprocess
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIGURATION ---
MODEL_ID = "google/gemma-3-12b-it"
PROMPTS_FILE = "data/prompts/EvaluationPrompts.xlsx"
OUTPUT_DIR = "outputs/generated_scripts_strict2"
LOGS_DIR = "reports/logs_strict2"
STL_DIR = "outputs/3d_models_strict2"
MAX_RETRIES = 10

def setup_environment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(STL_DIR, exist_ok=True)

def load_prompts():
    if not os.path.exists(PROMPTS_FILE):
        raise FileNotFoundError(f"Missing prompt file at {PROMPTS_FILE}.")
    
    df = pd.read_excel(PROMPTS_FILE)
    
    # Change "Prompts" below to the EXACT name of the column header in your Excel sheet 
    # that contains the actual text descriptions (e.g., "Geometry Description" or "Prompts")
    prompt_column_name = "Prompts" 
    
    if prompt_column_name not in df.columns:
        print(f"ERROR: Could not find '{prompt_column_name}'. Available columns are: {df.columns.tolist()}")
        sys.exit(1)
        
    prompt_list = df[prompt_column_name].dropna().tolist()
    
    prompts = []
    for i, text in enumerate(prompt_list):
        if str(text).strip() in ['-', '', 'nan']: continue
        prompts.append({"id": f"GEOM_{i+1:03d}", "prompt": str(text).strip()})
        
    return prompts

def initialize_model():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return tokenizer, model

def extract_python_code(llm_output):
    match = re.search(r"```python(.*?)```", llm_output, re.DOTALL)
    if match: return match.group(1).strip()
    return llm_output.strip()

def run_cadquery_strict(pure_code, stl_output_path):
    prepend_code = """import cadquery as cq
_cq_export_target = None
def show_object(obj, *args, **kwargs):
    global _cq_export_target
    _cq_export_target = obj
"""

    append_code = f"""
import sys
exported = False
try:
    if _cq_export_target is not None:
        cq.exporters.export(_cq_export_target, '{stl_output_path}')
        exported = True
    elif 'result' in locals() and isinstance(locals()['result'], (cq.Workplane, cq.Assembly, cq.Shape, cq.Compound, cq.Solid)):
        cq.exporters.export(locals()['result'], '{stl_output_path}')
        exported = True
    else:
        for var_name, var_val in list(locals().items()):
            if var_name.startswith('_'): continue
            if isinstance(var_val, (cq.Workplane, cq.Assembly, cq.Shape, cq.Compound, cq.Solid)):
                cq.exporters.export(var_val, '{stl_output_path}')
                exported = True
                break
                
    if not exported:
        print("SEMANTIC ERROR: No valid 3D CadQuery object was found to export.")
        sys.exit(1)
        
    print("SUCCESS_EXPORT")
except Exception as e:
    print(f"EXECUTION CRASH: {{str(e)}}")
    sys.exit(1)
"""
    
    temp_path = "temp_strict_eval.py"
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(prepend_code + "\n" + pure_code + "\n" + append_code)

    try:
        result = subprocess.run([sys.executable, temp_path], capture_output=True, text=True, timeout=30)
        if os.path.exists(temp_path): os.remove(temp_path)
        
        if result.returncode == 0 and "SUCCESS_EXPORT" in result.stdout:
            return True, "SUCCESS"
        else:
            error_msg = (result.stdout.strip() + "\n" + result.stderr.strip()).strip()
            return False, error_msg
            
    except subprocess.TimeoutExpired:
        if os.path.exists(temp_path): os.remove(temp_path)
        return False, "TimeoutExpired: Script execution took longer than 30 seconds."
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return False, str(e)

def generate_response(tokenizer, model, messages):
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    input_length = inputs["input_ids"].shape[1]
    response_ids = output_ids[0][input_length:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)

def main():
    setup_environment()
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} valid prompts.")
    tokenizer, model = initialize_model()
    
    # Strictly base system instruction matching your previous clean runs
    system_instruction = (
        "You are an expert CAD engineer. Output functional, mathematically precise Python "
        "scripts using the CadQuery framework. Return ONLY the raw python code wrapped in "
        "```python ``` markdown."
    )
    
    results_log = []
    
    for item in tqdm(prompts, desc="Strict Benchmarking"):
        prompt_id = item["id"]
        base_prompt = item["prompt"]
        
        script_path = os.path.join(OUTPUT_DIR, f"{prompt_id}.py")
        stl_path = os.path.join(STL_DIR, f"{prompt_id}.stl")
        log_path = os.path.join(LOGS_DIR, f"{prompt_id}_eval.log")
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": base_prompt}
        ]
        
        success = False
        final_error = ""
        attempts = 0
        
        while attempts <= MAX_RETRIES and not success:
            raw_response = generate_response(tokenizer, model, messages)
            pure_code = extract_python_code(raw_response)
            
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(pure_code)
                
            success, error_trace = run_cadquery_strict(pure_code, stl_path)
            
            if success:
                with open(log_path, "w") as f: f.write(f"Attempt {attempts + 1}: STRICT SUCCESS\n")
                break
            else:
                attempts += 1
                final_error = error_trace
                with open(log_path, "a") as f: f.write(f"--- Attempt {attempts} FAILED ---\n{error_trace}\n\n")
                
                messages.append({"role": "assistant", "content": f"```python\n{pure_code}\n```"})
                error_prompt = (
                    f"The script failed. Error:\n\n{error_trace}\n\n"
                    "Please fix the error. Return ONLY the corrected python code wrapped in ```python ``` markdown."
                )
                messages.append({"role": "user", "content": error_prompt})

        status = "Pass" if success else "Fail"
        results_log.append({
            "Prompt ID": prompt_id,
            "Original Prompt": base_prompt,
            "Status": status,
            "Attempts Needed": attempts if success else attempts,
            "Final Error": final_error if not success else "None"
        })

    summary_df = pd.DataFrame(results_log)
    summary_df.to_csv(os.path.join(LOGS_DIR, "evaluation_summary.csv"), index=False)
    print("\nStrict Benchmark complete! Check reports/logs_strict/evaluation_summary.csv")

if __name__ == "__main__":
    main()