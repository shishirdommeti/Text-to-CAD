import pandas as pd
import json

def export_to_json(excel_file, output_json):
    print(f"Reading {excel_file}...")
    df = pd.read_excel(excel_file)
    
    output_data = []
    success_count = 0
    
    for index, row in df.iterrows():
        if row['Valid_Syntax']:
            code = str(row['Generated_CAD_Code']).strip()
            
            output_data.append({
                "prompt_id": f"prompt_{index+1}", 
                "output": code                    
            })
            success_count += 1

    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Successfully packaged {success_count} sequences into '{output_json}'!")

export_to_json("CADFusion_Reasoning_Results.xlsx", "cad_outputs_v2.json")