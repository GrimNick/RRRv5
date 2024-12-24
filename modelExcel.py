import pandas as pd
import os
import sys
import subprocess

# Path of the input Excel file
input_path = sys.argv[1] 

# Extract directory and file name to construct the output path
directory, filename = os.path.split(input_path)
file_root, file_ext = os.path.splitext(filename)
output_path = os.path.join(directory, f"{file_root}2{file_ext}")

# Open the input Excel file
excel_data = pd.ExcelFile(input_path)

# Dictionary to store processed data frames for each sheet
output_data = {}

# Process each sheet in the input Excel file
for sheet_name in excel_data.sheet_names:
    print(f"Processing sheet: {sheet_name}", flush=True)  # Print and flush for real-time output
    
    # Load the sheet data into a DataFrame
    df = excel_data.parse(sheet_name)

    # Check if the necessary columns exist
    if 'Time' in df.columns and 'Velocity' in df.columns and 'Relative Velocity' in df.columns:
        # Interpolate missing velocities (NaN) using linear interpolation
        df['Velocity'] = df['Velocity'].interpolate(method='linear', limit_direction='forward', axis=0)
        
        # Check for duplicates in 'Time' column
        duplicates = df[df.duplicated('Time', keep=False)]
        
        if not duplicates.empty:
            print(f"Duplicates found in sheet '{sheet_name}' for Time values:", flush=True)
            print(duplicates[['Time', 'Velocity', 'Relative Velocity']], flush=True)

        # Group by 'Time' and average 'Velocity' and 'Relative Velocity' for duplicate Time values
        df_processed = (
            df.groupby('Time', as_index=False)
            .agg({
                'Velocity': 'mean',
                'Relative Velocity': 'mean'
            })
        )
    else:
        # If columns don't match, copy the sheet data as-is
        df_processed = df

    # Store the processed DataFrame in the output_data dictionary
    output_data[sheet_name] = df_processed

# Write the processed sheets to a new Excel file
with pd.ExcelWriter(output_path) as writer:
    for sheet_name, df_processed in output_data.items():
        df_processed.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"New Excel file created at: {output_path}", flush=True)
print("Processing complete", flush=True)

# Run the next step with the updated file path
subprocess.run([sys.executable, 'modelKmPerHour.py', output_path])  # You can modify the path of modelExcel.py as needed
