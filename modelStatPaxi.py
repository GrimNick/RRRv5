import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os 
import sys
model = load_model('hamroRRR.h5')

input_path = sys.argv[1]
new_all_sheets = pd.read_excel(input_path, sheet_name=None)

# Extract directory and file name to construct the output path
directory, filename = os.path.split(input_path)
file_root, file_ext = os.path.splitext(filename)
output_path = os.path.join(directory, f"{file_root[:-1]}6{file_ext}")
scaler = MinMaxScaler()

final_predictions = []

for sheet_name, df in new_all_sheets.items():
    print(f"Processing sheet: {sheet_name}")
    
    df[['Mean Velocity', 'Standard Deviation', 'Skewness', 'Kurtosis', 'Variance', 'Peak-to-Peak', 'Comparison Mean']] = scaler.fit_transform(
        df[['Mean Velocity', 'Standard Deviation', 'Skewness', 'Kurtosis', 'Variance', 'Peak-to-Peak', 'Comparison Mean']]
    )
    
    df_features = df.drop(columns=['Track ID'])

    X_new = df_features[['Mean Velocity', 'Standard Deviation', 'Skewness', 'Kurtosis', 'Variance', 'Peak-to-Peak', 'Comparison Mean']].values
    X_new = X_new.astype(np.float32)  # Ensure correct data type

    X_new = X_new.reshape((X_new.shape[0], 1, X_new.shape[1]))  # Reshape for RNN input (1 time step per sample)
    print(f"Shape of X_new after reshaping (samples, time_steps, features): {X_new.shape}")

    predictions = model.predict(X_new)
    
    predictions = (predictions > 0.5).astype(int)

    df_predictions = df[['Track ID']].copy()  # Retain Track ID only for output
    df_predictions['Predictions'] = predictions.flatten()  # Add predictions column

    final_predictions.append(df_predictions)

predictions_df = pd.concat(final_predictions, ignore_index=True)

print(predictions_df.head())
  # Write the combined data to the output Excel file
with pd.ExcelWriter(output_path) as writer:
 predictions_df.to_excel(writer, sheet_name=sheet_name, index=False)
