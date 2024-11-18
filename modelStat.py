import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import openpyxl

# Define input and output file paths
input_excel_path = r"E:\Videoo\track2_processed_data3.xlsx"
output_excel_path = r"E:\Videoo\track2_processed_data3_output.xlsx"

# Load all sheets from the original Excel file
sheets = pd.read_excel(input_excel_path, sheet_name=None, engine='openpyxl')

# Initialize the Excel writer with the output path
with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
    # Loop over each sheet
    for sheet_name, df in sheets.items():
        # Ensure Velocity and Relative Velocity columns are numerical
        df["Velocity"] = pd.to_numeric(df["Velocity"], errors='coerce')
        df["Relative Velocity"] = pd.to_numeric(df["Relative Velocity"], errors='coerce')

        # Filter out rows with missing values in Velocity or Relative Velocity columns
        df = df.dropna(subset=["Velocity", "Relative Velocity"])
        
        # Calculate parameters
        velocity_std = df["Velocity"].std()
        velocity_mean = df["Velocity"].mean()
        velocity_variance = df["Velocity"].var()
        velocity_skewness = skew(df["Velocity"])
        velocity_kurtosis = kurtosis(df["Velocity"])

        # Calculate peak-to-peak (max - min) for Velocity
        peak_to_peak = df["Velocity"].max() - df["Velocity"].min()

        # Calculate mean of absolute difference between Velocity and Relative Velocity
        comparison_mean = (df["Velocity"] - df["Relative Velocity"]).abs().mean()

        # Comments based on calculated values
        comments = {
            "Standard Deviation": "Low S.D. indicates consistent speed; high S.D. suggests fluctuation.",
            "Mean Velocity": f"A mean of {velocity_mean:.2f} km/h suggests typical vehicle speed behavior.",
            "Variance": f"Variance of {velocity_variance:.2f} shows the degree of speed spread.",
            "Skewness": f"Skewness of {velocity_skewness:.2f} shows tendency for acceleration or deceleration.",
            "Kurtosis": f"Kurtosis of {velocity_kurtosis:.2f} shows concentration around mean velocity.",
            "Comparison Mean": f"The mean absolute difference of {comparison_mean:.2f} km/h shows the average deviation between velocity and relative velocity.",
            "Peak-to-Peak": f"The peak-to-peak value of {peak_to_peak:.2f} km/h shows the range of velocity fluctuation."
        }

        # Prepare the calculated parameters as a DataFrame
        df_params = pd.DataFrame({
            "Parameter": ["Standard Deviation", "Mean Velocity", "Variance", "Skewness", "Kurtosis", "Comparison Mean", "Peak-to-Peak"],
            "Value": [velocity_std, velocity_mean, velocity_variance, velocity_skewness, velocity_kurtosis, comparison_mean, peak_to_peak],
            "Comment": list(comments.values())
        })

        # Insert calculated parameters after the "Relative Velocity" column
        df_combined = pd.concat([df, df_params], axis=1)

        # Write the combined data to the output Excel file
        df_combined.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Calculated parameters and comments have been saved to the output Excel file.")
