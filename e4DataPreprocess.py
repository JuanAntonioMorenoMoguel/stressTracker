import pandas as pd
import numpy as np

# Define a function to process each file
def process_file(filename):
    filepath = '/kaggle/input/smoke1/' + filename  # Adjust as necessary for your environment
    with open(filepath, 'r') as f:
        # Read and attempt to parse the first row
        first_row = f.readline().strip()
        try:
            initial_timestamp = float(first_row.split(',')[0])  # Take the first value only
        except ValueError:
            raise ValueError(f"Unexpected format in the first row of {filename}: {first_row}")
        
        # Read and attempt to parse the second row
        second_row = f.readline().strip()
        try:
            sample_rate = float(second_row.split(',')[0])  # Take the first value only
        except ValueError:
            sample_rate = None  # Allow for cases where sample rate is missing
        
    # Read the rest of the data
    data = pd.read_csv(filepath, skiprows=2, header=None)
    return initial_timestamp, sample_rate, data

# List of files to process (excluding IBI.csv)
file_names = ["TEMP.csv", "EDA.csv", "BVP.csv", "ACC.csv", "HR.csv"]

# Process each file and store their details
file_data = {}
for file in file_names:
    initial_timestamp, sample_rate, data = process_file(file)
    file_data[file] = {
        "timestamp": initial_timestamp,
        "sample_rate": sample_rate,
        "data": data
    }

# Find the file with the highest frequency
reference_file = max(
    file_data.keys(),
    key=lambda x: file_data[x]["sample_rate"] if file_data[x]["sample_rate"] else 0
)
reference_rate = file_data[reference_file]["sample_rate"]
reference_timestamp = file_data[reference_file]["timestamp"]

# Resample all data to align with the reference frequency
master_data = pd.DataFrame()
for file, details in file_data.items():
    data = details["data"]
    timestamp = details["timestamp"]
    sample_rate = details["sample_rate"]

    if sample_rate is not None:
        # Calculate time column
        time_index = np.arange(0, len(data)) / sample_rate + (timestamp - reference_timestamp)
        data["time"] = time_index
        # Interpolate to match reference frequency
        resampled_time = np.arange(0, time_index[-1], 1 / reference_rate)
        data = data.set_index("time").reindex(resampled_time).interpolate("linear").reset_index()
        data.rename(columns={"index": "time"}, inplace=True)

        # Rename columns based on the file
        if file == "TEMP.csv":
            data.columns = ["time", "TEMP"]
        elif file == "EDA.csv":
            data.columns = ["time", "EDA"]
        elif file == "BVP.csv":
            data.columns = ["time", "BVP"]
        elif file == "ACC.csv":
            data.columns = ["time", "ACC_x", "ACC_y", "ACC_z"]
        elif file == "HR.csv":
            data.columns = ["time", "HR"]
    
        # Merge with master_data
        if master_data.empty:
            master_data = data
        else:
            master_data = pd.merge(master_data, data, on="time", how="outer")

# Sort the master data by time
master_data.sort_values("time", inplace=True)

master_data = master_data.dropna()

# Export to CSV
master_data.to_csv("master.csv", index=False)
print("Master CSV created: master.csv")
