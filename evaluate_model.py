import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
from train_pipeline import ModifiedResNet18  # Make sure this matches your model definition

MODEL_PATH = "best_model.pth"
TEST_CSV = "test.csv"
IMAGE_DIRS = ["For Lee/pageOne/", "For Lee/pageTwo/", "For Lee/pageThree/"]
LABELS = ['healthy', 'sb']

# Load the trained binary classifier
model = ModifiedResNet18(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_data = pd.read_csv(TEST_CSV)
# Only keep 'healthy' and 'sb'
test_data = test_data[test_data['disease'].isin(['healthy', 'sb'])].reset_index(drop=True)
print(f"Filtered test data size (healthy/sb only): {len(test_data)}")

# Helper function to find image path
def find_image_path(filename, image_dirs):
    for image_dir in image_dirs:
        img_path = os.path.join(image_dir, filename)
        if os.path.exists(img_path):
            return img_path
    raise FileNotFoundError(f"Image {filename} not found in any of the directories.")

# Loss function
criterion = nn.CrossEntropyLoss()

# Evaluation loop
for idx, row in test_data.iterrows():
    filename = row['filename']
    actual_label = row['disease']
    print(f"Processing file: {filename}")

    try:
        img_path = find_image_path(filename, IMAGE_DIRS)
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_idx = torch.argmax(outputs, dim=1)
            predicted_label = LABELS[predicted_idx.item()]
            actual_idx = 0 if actual_label == 'healthy' else 1
            loss = criterion(outputs, torch.tensor([actual_idx]))

        print(f"Actual: {actual_label}, Predicted: {predicted_label}, Loss: {loss.item():.4f}")

        plt.imshow(image)
        plt.title(f'Actual: {actual_label} | Predicted: {predicted_label} | Loss: {loss.item():.4f}')
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"Error processing {filename}: {e}")