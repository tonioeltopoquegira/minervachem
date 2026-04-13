import pandas as pd
import numpy as np
import os

# Get the directory where this module is located
_module_dir = os.path.dirname(os.path.abspath(__file__))
_static_dir = os.path.join(os.path.dirname(os.path.dirname(_module_dir)), 'static')

# Load QM9 dataset
df = pd.read_csv(os.path.join(_static_dir, 'qm9_processed.csv'))

# Define the target property columns
property_cols = ['E_at', 'zpve', 'e_gap', 'C_v']

# Filter out rows with missing values
df_valid = df.dropna(subset=property_cols)

# Compute min and max for each property
min_values = df_valid[property_cols].min()
max_values = df_valid[property_cols].max()

# Combine into a DataFrame
range_df = pd.DataFrame({
    'target': property_cols,
    'min': min_values.values,
    'max': max_values.values
})

# Save to CSV
output_path = 'qm9_target_ranges.csv'
range_df.to_csv(output_path, index=False)

# Print results
print("Target ranges (based on full QM9 dataset):\n")
for _, row in range_df.iterrows():
    print(f"{row['target']:6s} | min: {row['min']:.6f} | max: {row['max']:.6f}")

print(f"\nSaved to: {output_path}")
