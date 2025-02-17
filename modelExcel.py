import pandas as pd
import os
import sys
import subprocess
import time

# Ensure at least one argument (input file path) is provided
if len(sys.argv) < 2:
    print("Error: No input file provided.", flush=True)
    sys.exit(1)

# Path of the input Excel file
input_path = sys.argv[1]

# Verify if the file exists before proceeding
if not os.path.exists(input_path):
    print(f"Error: File '{input_path}' not found.", flush=True)
    sys.exit(1)

# Extract directory and file name to construct the output path
directory, filename = os.path.split(input_path)
file_root, file_ext = os.path.splitext(filename)
output_path = os.path.join(directory, f"{file_root}_processed{file_ext}")

# Open the input Excel file with explicit `engine="openpyxl"`
try:
    excel_data = pd.ExcelFile(input_path, engine="openpyxl")  # FIXED HERE
except Exception as e:
    print(f" Error opening Excel file: {e}", flush=True)
    sys.exit(1)

# Dictionary to store processed data frames for each sheet
output_data = {}

# Number of sheets to process (for progress updates)
total_sheets = len(excel_data.sheet_names)
processed_sheets = 0

if total_sheets == 0:
    print("Error: The provided Excel file has no sheets.", flush=True)
    sys.exit(1)

# Process each sheet in the input Excel file
for sheet_name in excel_data.sheet_names:
    print(f"Processing sheet: {sheet_name}", flush=True)  # Print and flush for real-time output
    
    # Load the sheet data into a DataFrame
    try:
        df = excel_data.parse(sheet_name, engine="openpyxl")  # ✅ FIXED HERE
    except Exception as e:
        print(f"Error reading sheet '{sheet_name}': {e}", flush=True)
        continue

    if df.empty:
        print(f"⚠️ Warning: Sheet '{sheet_name}' is empty. Skipping processing.", flush=True)
        continue

    # Check if required columns exist
    required_columns = {'Time', 'Velocity', 'Relative Velocity'}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        print(f"⚠️ Warning: Missing columns {missing_columns} in sheet '{sheet_name}'. Copying data as-is.", flush=True)
        df_processed = df  # Keep original data if necessary columns are missing
    else:
        # Check for duplicate 'Time' values
        duplicates = df[df.duplicated('Time', keep=False)]
        if not duplicates.empty:
            print(f"⚠️ Warning: Duplicates found in sheet '{sheet_name}' for 'Time' column:", flush=True)
            print(duplicates[['Time', 'Velocity', 'Relative Velocity']], flush=True)

        # Group by 'Time' and average 'Velocity' and 'Relative Velocity'
        df_processed = (
            df.groupby('Time', as_index=False)
            .agg({
                'Velocity': 'mean',
                'Relative Velocity': 'mean'
            })
        )

    # Store processed DataFrame
    output_data[sheet_name] = df_processed

    # **Emit progress updates (Simulating Electron IPC)**
    processed_sheets += 1
    progress_percentage = int((processed_sheets / total_sheets) * 100)
    print(f" Progress: {progress_percentage}%", flush=True)  # **Ensure Electron receives updates**

    # **Simulate processing delay for better visibility (remove in actual use)**
    time.sleep(0.5)

# Write processed sheets to a new Excel file
try:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:  # ✅ FIXED HERE
        for sheet_name, df_processed in output_data.items():
            df_processed.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f" New Excel file created at: {output_path}", flush=True)
except Exception as e:
    print(f"Error writing output file: {e}", flush=True)
    sys.exit(1)

print("Processing complete!", flush=True)

# **Run `modelKmPerHour.py` on processed Excel**
try:
    subprocess.run([sys.executable, 'modelKmPerHour.py', output_path], check=True)
    print("Successfully executed modelKmPerHour.py", flush=True)
except subprocess.CalledProcessError as e:
    print(f" Error running modelKmPerHour.py: {e}", flush=True)
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error while running modelKmPerHour.py: {e}", flush=True)
    sys.exit(1)
