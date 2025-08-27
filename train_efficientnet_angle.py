import os
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import default_collate

IMAGE_DIRS = ["For Lee/pageOne/", "For Lee/pageTwo/", "For Lee/pageThree/"]
TRAIN_CSV = "train_angle.csv"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
EFFNET_WEIGHTS_PATH = "best_efficientnet_angle.pth"
LABELS = ['top', 'side']  # Updated to include only 'top' and 'side'

class ZebrafishDataset(Dataset):
    def __init__(self, csv_file=None, image_dirs=None, transform=None, data=None):
        if csv_file:
            self.data = pd.read_csv(csv_file)
            # Only keep 'top' and 'side'
            self.data = self.data[self.data['angle'].isin(LABELS)].reset_index(drop=True)
        elif data is not None:
            self.data = data
        else:
            raise ValueError("Either csv_file or data must be provided.")
        self.image_dirs = image_dirs
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        filename = self.data.iloc[idx, 0]
        for image_dir in self.image_dirs:
            img_path = os.path.join(image_dir, filename)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                break
        else:
            print(f"Warning: Image {filename} not found in any of the directories. Skipping.")
            return None, None
        label = self.data.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if image is None or label is None:
            return None
        return image, label

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

if __name__ == "__main__":
    dataset = ZebrafishDataset(csv_file=TRAIN_CSV, image_dirs=IMAGE_DIRS, transform=transform)
    train_data, val_data = train_test_split(dataset.data, test_size=0.2, random_state=42)
    train_dataset = ZebrafishDataset(image_dirs=IMAGE_DIRS, transform=transform, data=train_data.reset_index(drop=True))
    val_dataset = ZebrafishDataset(image_dirs=IMAGE_DIRS, transform=transform, data=val_data.reset_index(drop=True))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # Updated for binary classification

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            labels = torch.tensor([0 if label == 'top' else 1 for label in labels])  # Map 'top' to 0 and 'side' to 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                labels = torch.tensor([0 if label == 'top' else 1 for label in labels])  # Map 'top' to 0 and 'side' to 1
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), EFFNET_WEIGHTS_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    print("Training complete.")
    model.load_state_dict(torch.load(EFFNET_WEIGHTS_PATH))
    model.eval()
    test_dataset = ZebrafishDataset(csv_file="test_angle.csv", image_dirs=IMAGE_DIRS, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    def evaluate_model(model, test_loader):
        correct = 0
        total = 0
        model.to(device)
        with torch.no_grad():
            for images, labels in test_loader:
                labels = torch.tensor([0 if label == 'top' else 1 for label in labels])  # Map 'top' to 0 and 'side' to 1
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
    evaluate_model(model, test_loader)