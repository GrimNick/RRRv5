import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

model = load_model('hamroRRR.h5')

input_file = "E://Videoo//track6_processed_data5.xlsx" #shila milau
new_all_sheets = pd.read_excel(input_file, sheet_name=None)

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

predictions_df.to_excel('E://Videoo//track6_processed_data6.xlsx', index=False)
