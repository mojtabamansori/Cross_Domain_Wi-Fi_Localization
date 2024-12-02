import pandas as pd

# Load the original CSV
df = pd.read_csv("annotation.csv")

# Select the last 6 columns
last_six_cols = df.columns[-6:]

# Filter rows: one column should be "nothing" and all others should not be empty ("")
filtered_df = df[last_six_cols].apply(
    lambda row: (row == 'nothing').sum() == 1 and (row != '').all(), axis=1
)

# Create a new DataFrame with the filtered rows
new_df = df[filtered_df]

# Save the filtered rows to a new CSV file
new_df.to_csv("filtered_annotation.csv", index=False)

print("Filtered rows saved to 'filtered_annotation.csv'.")
