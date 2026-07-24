import pandas as pd
import json
import os

# Updated for the Reasoning Benchmark
excel_file = "CADFusion_Reasoning_Results.xlsx"
json_file = "cad_outputs_v2.json"
objects_folder = "./visual_objects_v2"
output_excel = "FINAL_Mapped_Reasoning_Results.xlsx"

print(f"Reading JSON to map reasoning folders back to original prompts...")

# Load the Rosetta Stone (JSON)
with open(json_file, "r") as f:
    json_data = json.load(f)

# Load your Excel file
df = pd.read_excel(excel_file)

# Create two new columns for your final grading
df['Passed_Compiler'] = False
df['3D_Folder_Name'] = ""

# Loop through the Microsoft folders
success_count = 0
for idx, item in enumerate(json_data):
    folder_name = str(idx).zfill(6) # e.g., "000000"
    folder_path = os.path.join(objects_folder, folder_name)
    
    # Did Microsoft successfully generate a 3D file in this folder?
    if os.path.exists(folder_path) and len(os.listdir(folder_path)) > 0:
        
        # Get the original Excel row index (e.g., "prompt_43" -> 42)
        original_row_index = int(item['prompt_id'].split('_')[1]) - 1
        
        # Update the spreadsheet!
        df.at[original_row_index, 'Passed_Compiler'] = True
        df.at[original_row_index, '3D_Folder_Name'] = folder_name
        success_count += 1

# Save the perfectly mapped Excel file
df.to_excel(output_excel, index=False)
print(f"Mapping complete! Found {success_count} successful reasoning geometries.")
print(f"Saved to {output_excel}")