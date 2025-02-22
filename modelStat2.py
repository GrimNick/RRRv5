import pandas as pd
import openpyxl
import os 
import sys
import subprocess

# Define input and output file paths
input_excel_path = sys.argv[1]
# Extract directory and file name to construct the output path
directory, filename = os.path.split(input_excel_path)
file_root, file_ext = os.path.splitext(filename)
output_summary_path = os.path.join(directory, f"{file_root[:-1]}5{file_ext}")


# Load all sheets from the input Excel file
sheets = pd.read_excel(input_excel_path, sheet_name=None, engine='openpyxl')

# Initialize a list to store summary data
summary_data = []

# Loop over each sheet
for sheet_name, df in sheets.items():
    # Extract parameters from the Parameter and Value columns
    track_id = sheet_name  # Use the sheet name as the Track ID
    parameters = df[["Parameter", "Value"]].dropna()  # Drop rows with NaN values

    # Define a helper function to retrieve the parameter value or None if not found
    def get_parameter_value(param_name):
        value = parameters.loc[parameters["Parameter"] == param_name, "Value"]
        return value.values[0] if not value.empty else None

    # Extract each calculated value based on the Parameter column
    std_dev = get_parameter_value("Standard Deviation")
    mean_velocity = get_parameter_value("Mean Velocity")
    variance = get_parameter_value("Variance")
    skewness = get_parameter_value("Skewness")
    kurtosis = get_parameter_value("Kurtosis")
    peak_to_peak = get_parameter_value("Peak-to-Peak")
    comparison_mean = get_parameter_value("Comparison Mean")

    # Replace NaN values with 0
    std_dev = 0 if pd.isna(std_dev) else std_dev
    mean_velocity = 0 if pd.isna(mean_velocity) else mean_velocity
    variance = 0 if pd.isna(variance) else variance
    skewness = 0 if pd.isna(skewness) else skewness
    kurtosis = 0 if pd.isna(kurtosis) else kurtosis
    peak_to_peak = 0 if pd.isna(peak_to_peak) else peak_to_peak
    comparison_mean = 0 if pd.isna(comparison_mean) else comparison_mean

    # Set flags based on specified conditions
    flag_std_dev = 1 if std_dev > 15 else 0
    flag_mean_velocity = 1 if mean_velocity > 40 else 0
    flag_variance = 1 if variance > 150 else 0
    flag_skewness = 1 if skewness > 6 else 0
    flag_kurtosis = 1 if kurtosis < 10 else 0
    flag_peak_to_peak = 1 if peak_to_peak > 40 else 0
    flag_comparison_mean = 1 if comparison_mean > 10 else 0
    # Calculate total flag as the sum of all individual flags
    total_flag = (
        flag_std_dev + flag_mean_velocity + flag_variance +
        flag_skewness + flag_kurtosis + flag_peak_to_peak + flag_comparison_mean
    )
    flag_reckless_status = 'Yes' if total_flag >= 4 else 'No'
    
    # Add all values to the summary entry
    summary_entry = {
        "Track ID": track_id,
        "Reckless Status": flag_reckless_status,
        "Total Flag": total_flag,
        "Standard Deviation": std_dev,
        "Flag Standard Deviation": flag_std_dev,
        "Mean Velocity": mean_velocity,
        "Flag Mean Velocity": flag_mean_velocity,
        "Variance": variance,
        "Flag Variance": flag_variance, 
        "Skewness": skewness,
        "Flag Skewness": flag_skewness,
        "Kurtosis": kurtosis,
        "Flag Kurtosis": flag_kurtosis,
        "Peak-to-Peak": peak_to_peak,
        "Flag Peak-to-Peak": flag_peak_to_peak,
        "Comparison Mean": comparison_mean,
        "Flag Comparison Mean": flag_comparison_mean,
    }
    
    # Append the entry to the summary list
    summary_data.append(summary_entry)

# Create a DataFrame for the summary data
summary_df = pd.DataFrame(summary_data)

# Sort the DataFrame by Total Flag in descending order
summary_df = summary_df.sort_values(by="Total Flag", ascending=False)

# Save the sorted summary to a new Excel file
with pd.ExcelWriter(output_summary_path, engine='openpyxl') as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

# Reopen the output Excel file to adjust column widths
workbook = openpyxl.load_workbook(output_summary_path)
worksheet = workbook["Summary"]

# Set each column width to 140 pixels (20 Excel units)
for column in worksheet.columns:
    column_letter = column[0].column_letter  # Get the column letter
    worksheet.column_dimensions[column_letter].width = 20

# Save the workbook with updated column widths
workbook.save(output_summary_path)

print("Sorted summary with flags has been saved to the output Excel file with adjusted column widths.")


subprocess.run([sys.executable, 'modelStatPaxi.py',  output_summary_path]) 