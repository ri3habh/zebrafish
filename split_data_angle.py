import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV files
data1 = pd.read_csv('pageOneAngle.csv')
data2 = pd.read_csv('pageTwoAngle.csv')
data3 = pd.read_csv('pageThreeAngle.csv')

# Combine the data into a single DataFrame
data = pd.concat([data1, data2, data3], ignore_index=True)

# Shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the data into training (70%), validation (15%), and test (15%)
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Save the splits into separate CSV files
train_data.to_csv('train_angle.csv', index=False)
validation_data.to_csv('validation_angle.csv', index=False)
test_data.to_csv('test_angle.csv', index=False)

print("Data has been split and saved into train_angle.csv, validation_angle.csv, and test_angle.csv.")