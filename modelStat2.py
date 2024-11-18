import pandas as pd
import openpyxl

# Define input and output file paths
input_excel_path = r"E:\Videoo\track2_processed_data3_output.xlsx"
output_summary_path = r"E:\Videoo\track2_summary_with_flags_sorted.xlsx"

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

    # Set flags based on specified conditions
    flag_std_dev = 1 if std_dev and std_dev > 15 else 0
    flag_mean_velocity = 1 if mean_velocity and mean_velocity > 40 else 0
    flag_variance = 1 if variance and variance > 150 else 0
    flag_skewness = 1 if skewness and skewness > 6 else 0
    flag_kurtosis = 1 if kurtosis and kurtosis < 10 else 0
    flag_peak_to_peak = 1 if peak_to_peak and peak_to_peak > 40 else 0
    flag_comparison_mean = 1 if comparison_mean and comparison_mean > 10 else 0
    
    # Calculate total flag as the sum of all individual flags
    total_flag = (
        flag_std_dev + flag_mean_velocity + flag_variance +
        flag_skewness + flag_kurtosis + flag_peak_to_peak + flag_comparison_mean
    )

    # Add all values to the summary entry
    summary_entry = {
        "Track ID": track_id,
        "Standard Deviation": std_dev,
        "Mean Velocity": mean_velocity,
        "Variance": variance,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "Peak-to-Peak": peak_to_peak,
        "Comparison Mean": comparison_mean,
        "Flag Standard Deviation": flag_std_dev,
        "Flag Mean Velocity": flag_mean_velocity,
        "Flag Variance": flag_variance,
        "Flag Skewness": flag_skewness,
        "Flag Kurtosis": flag_kurtosis,
        "Flag Peak-to-Peak": flag_peak_to_peak,
        "Flag Comparison Mean": flag_comparison_mean,
        "Total Flag": total_flag
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
