import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

# LOAD MODELS
print("Loading Base LLaMA Model (This may take a few minutes)...")
base_model_id = "NousResearch/Meta-Llama-3-8B" 

# Load the base tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16).cuda()

base_model.resize_token_embeddings(128258)

print("Applying Microsoft's CADFusion Patch...")
model = PeftModel.from_pretrained(base_model, "microsoft/CADFusion", subfolder="v1_1")

# LOAD REASONING EXCEL FILE
input_excel = "EvaluationPrompts(Reasoning).xlsx" 
output_excel = "CADFusion_Reasoning_Results.xlsx"

# Load the file
df = pd.read_excel(input_excel)

# Create columns for our benchmark results
df['Generated_CAD_Code'] = ""
df['Execution_Time_Seconds'] = 0.0
df['Valid_Syntax'] = False

# Evaluation loop
for index, row in df.iterrows():
    prompt = row['Prompts']
    print(f"\nProcessing Prompt {index + 1}/{len(df)}: {prompt}")
    
    # Appending the Chain of Thought trigger to the prompt
    input_text = f"Instruction: Generate a CAD model for the following description: {prompt} Let's think step by step.\nOutput:"
    
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    start_time = time.time()
    
    try:
        # INCREASED TOKENS TO 4096
        outputs = model.generate(**inputs, max_new_tokens=4096, pad_token_id=tokenizer.eos_token_id)
        
        # KEEP SPECIAL TOKENS BUT MANUALLY STRIP LLAMA TAGS
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=False)
        generated_code = generated_code.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").replace("<eos>", "").replace("<|begin_of_text|>", "").strip()
        
        if "Output:" in generated_code:
            generated_code = generated_code.split("Output:")[-1].strip()
        
        execution_time = time.time() - start_time
        
        # Basic Syntax Check (Still looks for the lowercase CADFusion tags)
        is_valid = "<extrude_end>" in generated_code or "<sketch_end>" in generated_code
        
        # Save results to the dataframe
        df.at[index, 'Generated_CAD_Code'] = generated_code
        df.at[index, 'Execution_Time_Seconds'] = execution_time
        df.at[index, 'Valid_Syntax'] = is_valid
        
        print(f"Success! Generated in {execution_time:.2f} seconds.")
        
    except Exception as e:
        print(f"Error generating model for prompt {index}: {e}")
        df.at[index, 'Generated_CAD_Code'] = "ERROR"

# Save directly back to a new Excel file
df.to_excel(output_excel, index=False)
print(f"\nBenchmark complete. Results saved to {output_excel}")