import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV files
data1 = pd.read_csv('pageOne.csv')
data2 = pd.read_csv('pageTwo.csv')
data3 = pd.read_csv('pageThree.csv')

# Combine the data into a single DataFrame
data = pd.concat([data1, data2, data3], ignore_index=True)

# Shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the data into training (70%), validation (15%), and test (15%)
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Save the splits into separate CSV files
train_data.to_csv('train.csv', index=False)
validation_data.to_csv('validation.csv', index=False)
test_data.to_csv('test.csv', index=False)

print("Data has been split and saved into train.csv, validation.csv, and test.csv.")