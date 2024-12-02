import pandas as pd
import numpy as np

# Load the data from CSV
df = pd.read_csv("filtered_annotation.csv")

# Extract the last 12 to 6 columns for user activity analysis
user_activity = df.iloc[:, -12:-6]
not_nan_positions = np.argwhere(~user_activity.isna().values)

# Get label locations where user activity is not NaN
label_location_indices = not_nan_positions[:, :]
user_activity_array = np.array(user_activity)
label_location = user_activity_array[label_location_indices[:, 0], label_location_indices[:, 1]]

# Extract other necessary columns
mat_name = df.iloc[:, 1]  # Second column, assumed to contain mat names
envs = df.iloc[:, 2]  # Third column, environment data
envs_numeric, unique_values = pd.factorize(envs)  # Convert envs to numeric labels
wifi_band = df.iloc[:, 3]  # Fourth column, Wi-Fi band information

# Extract last 6 columns for user name identification
user_activity_final = df.iloc[:, -6:]
not_nan_user_positions = np.argwhere(~user_activity_final.isna().values)
user_name_indices = not_nan_user_positions[:, 1]  # Column indices for user names

# Combine all required data into a DataFrame
combined_data = pd.DataFrame({
    'mat_name': mat_name,
    'envs_numeric': envs_numeric,
    'wifi_band': wifi_band,
    'user_name': user_name_indices,
    'label_location': label_location
})

# Save the combined data to an Excel file
combined_data.to_excel("output_combined_data.xlsx", index=False)

print("Data has been successfully saved to 'output_combined_data.xlsx'.")
