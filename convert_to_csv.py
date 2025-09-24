import pandas as pd
import os

# The path to your folder containing the Excel files.
# The 'r' before the string is important; it tells Python to treat backslashes as regular characters.
folder_path = r'C:\uae-stock-dashboard\data'

# Get a list of all files in the folder
try:
    all_files = os.listdir(folder_path)
    print(f"Found {len(all_files)} files in the directory.")
except FileNotFoundError:
    print(f"Error: The folder '{folder_path}' was not found. Please check the path.")
    exit()

# Loop through each file in the folder
for file_name in all_files:
    # Check if the file is an Excel file
    if file_name.endswith('.xlsx'):
        # Create the full path to the Excel file
        excel_file_path = os.path.join(folder_path, file_name)
        
        # Create the name for the new CSV file
        csv_file_name = file_name.replace('.xlsx', '.csv')
        csv_file_path = os.path.join(folder_path, csv_file_name)
        
        try:
            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(excel_file_path)
            
            # Save the DataFrame to a CSV file
            # index=False prevents pandas from writing row numbers into the CSV
            df.to_csv(csv_file_path, index=False)
            
            print(f"Successfully converted '{file_name}' to '{csv_file_name}'")
        
        except Exception as e:
            print(f"Could not convert {file_name}. Error: {e}")

print("\nConversion process finished! ✅")