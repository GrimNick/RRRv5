import pandas as pd
import os
import sys
import subprocess

# Path of the input Excel file (the one with '2' in its name)
input_path = sys.argv[1]

# Extract directory and file name to construct the output path
directory, filename = os.path.split(input_path)
file_root, file_ext = os.path.splitext(filename)

# Replace '2' with '3' in the filename
output_path = os.path.join(directory, f"{file_root[:-1]}3{file_ext}")    

# Constants for the road dimensions in meters
horizontal_road_length = 8.41 # in meters
vertical_road_length = 68.3    # in meters

# Open the input Excel file
excel_data = pd.ExcelFile(input_path)

# Dictionary to store the data frames
output_data = {}

# Process each sheet in the input Excel file
for sheet_name in excel_data.sheet_names:
    # Load the sheet data into a DataFrame
    df = excel_data.parse(sheet_name)

    # Check if the DataFrame has 5 or more rows
    if len(df) < 3:
        # Skip sheets with fewer than 5 rows
        print(f"Sheet '{sheet_name}' has less than 5 rows and will be skipped.")
        continue

    # Check if the required columns exist
    if 'Time' in df.columns and 'Velocity' in df.columns and 'Relative Velocity' in df.columns:
        
        # Handle division by zero by checking if max value is 0
        max_velocity = df['Velocity'].max()
        max_relative_velocity = df['Relative Velocity'].max()

        if max_velocity > 0:
            # Convert 'Velocity' from pixels to meters (horizontal scaling)
            horizontal_scale = horizontal_road_length / max_velocity
        else:
            horizontal_scale = 0  # Avoid division by zero, set a default scale

        if max_relative_velocity > 0:
            # Convert 'Relative Velocity' from pixels to meters (vertical scaling)
            vertical_scale = vertical_road_length / max_relative_velocity
        else:
            vertical_scale = 0  # Avoid division by zero, set a default scale
        
        # Apply scaling if scaling factors are non-zero
        if horizontal_scale > 0:
            df['Velocity'] = df['Velocity'] * horizontal_scale
        
        if vertical_scale > 0:
            df['Relative Velocity'] = df['Relative Velocity'] * vertical_scale

        # Convert from meters per second to kilometers per hour (1 m/s = 3.6 km/h)
        df['Velocity'] = df['Velocity'] * 3.6
        df['Relative Velocity'] = df['Relative Velocity'] * 3.6

    # Store the DataFrame in the output_data dictionary
    output_data[sheet_name] = df

# Write the data to a new Excel file with '3' in the name
with pd.ExcelWriter(output_path) as writer:
    for sheet_name, df in output_data.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"New Excel file created at: {output_path}")

subprocess.run([sys.executable, 'modelStat.py', output_path]) # You can modify the path of modelExcel.py as needed
