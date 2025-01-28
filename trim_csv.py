import pandas as pd

# Read the CSV file
# Replace 'input.csv' with your CSV file name
df = pd.read_csv('job-categorisation.csv')

# Take first 1000 rows
df_trimmed = df.head(100)

# Save the trimmed dataframe to a new CSV file
# Replace 'output_trimmed.csv' with your desired output file name
df_trimmed.to_csv('job-categorisation_100.csv', index=False)

print("CSV file has been trimmed to 1000 rows and saved successfully!")