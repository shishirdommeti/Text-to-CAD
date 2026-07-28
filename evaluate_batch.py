import torch
from transformers import pipeline
import re
import traceback
import pandas as pd
from huggingface_hub import hf_hub_download, login
from tqdm import tqdm
import cadquery as cq
import os

def extract_python_code(llm_response: str) -> str:
    """Extracts raw Python code from the LLM's markdown formatting."""
    match = re.search(r"```python\n(.*?)\n```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return llm_response.strip()

def evaluate_batch_baseline():
    # 1. Authenticate and Load Data
    hf_token = "YOUR HUGGING FACE TOKEN" #Replace
    login(token=hf_token)

    output_dir = "generated_cad_models"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading Text2CAD dataset...")
    csv_path = hf_hub_download(
        repo_id="SadilKhan/Text2CAD", 
        repo_type="dataset", 
        filename="text2cad_v1.1/text2cad_v1.1.csv",
        token=hf_token
    )
    df = pd.read_csv(csv_path)
    
    # Grab the first 100 rows for the baseline test
    test_df = df.head(100)
    
    # 2. Initialize Gemma 3 12B
    print("Loading Gemma 3 12B Instruct (Stable CUDA 12.1)...")
    pipe = pipeline(
        "text-generation",
        model="google/gemma-3-12b-it",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    system_prompt = "You are an expert CAD engineer. Write a Python script using the CadQuery library to build the exact geometry requested. Assign the final 3D solid to a variable named 'result'. Return ONLY the raw python code wrapped in ```python ``` markdown."
    
    # 3. Setup Metric Tracking
    results = {
        "total": len(test_df),
        "syntax_crashes": 0,
        "semantic_failures": 0,
        "phantom_passes": 0,
        "valid_solids": 0
    }

    print(f"\nStarting evaluation of {results['total']} prompts...\n")
    
    # 4. The Evaluation Loop using tqdm for a progress bar
    for index, row in tqdm(test_df.iterrows(), total=results['total']):
        # CHANGE THIS FOR DIFFERENT LEVEL PROMPTS
        prompt_text = str(row.get('expert', ''))
        
        # Skip empty rows if any exist
        if not prompt_text or prompt_text.lower() == 'nan':
            results["total"] -= 1
            continue
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text}
        ]

        try:
            # Generate the response
            outputs = pipe(messages, max_new_tokens=512, truncation=True)
            llm_response = outputs[0]["generated_text"][-1]["content"]
            generated_code = extract_python_code(llm_response)
            
            # Sandbox Execution
            sandbox_env = {}
            
            # Check 1: Syntactic Success (Compilation)
            exec(generated_code, sandbox_env)
            
            # Check 2: Semantic Success (Variable Assignment)
            if 'result' not in sandbox_env:
                results["semantic_failures"] += 1
                continue
                
            cad_object = sandbox_env['result']
            
            # Check 3: Geometric Fidelity (Is it a real 3D solid?)
            try:
                generated_volume = cad_object.val().Volume()
                if generated_volume <= 0.0:
                    results["phantom_passes"] += 1
                else:
                    results["valid_solids"] += 1
                    
                    # NEW: Save the successful model to your hard drive!
                    # We use the dataset row index and UID for easy tracking later
                    model_uid = row.get('uid', 'unknown').replace('/', '_')
                    filename = f"{output_dir}/row_{index}_{model_uid}.stl"
                    
                    # Export the geometry as an STL file
                    cq.exporters.export(cad_object, filename)
            except Exception:
                # If CadQuery fails to calculate a volume, it's a corrupted/empty object
                results["phantom_passes"] += 1
                
        except Exception:
            # If exec() fails, or CadQuery throws an API error during construction
            results["syntax_crashes"] += 1
            print("\n" + "="*40)
            print("🚨 FATAL COMPILER ERROR:")
            print(e)
            print("-" * 40)
            print("📝 WHAT THE MODEL WROTE:")
            print(generated_code) # or whatever your variable for the LLM output is named
            print("="*40)
            break # Stop the loop after the very first crash!

    # 5. Print Final Report
    print("\n" + "="*45)
    print(" 🏆 BASELINE EVALUATION RESULTS")
    print("="*45)
    print(f" Total Evaluated:    {results['total']}")
    print(f" Syntax Crashes:     {results['syntax_crashes']} (Failed to compile/API error)")
    print(f" Semantic Failures:  {results['semantic_failures']} (No 'result' variable)")
    print(f" Phantom Passes:     {results['phantom_passes']} (0 Volume / Empty Geometry)")
    print(f" Valid Solids:       {results['valid_solids']} (Successfully built a 3D shape)")
    
    success_rate = (results['valid_solids'] / results['total']) * 100
    print("-" * 45)
    print(f" 🎯 ZERO-SHOT SUCCESS RATE: {success_rate:.1f}%")
    print("="*45)

if __name__ == "__main__":
    evaluate_batch_baseline()