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

# remove misssing
# and remove all other "disease" values that are not healthy or sb

# change this so we either predict two classes:
# 1) healthy
# 2) sb 

# Define constants
IMAGE_DIRS = ["For Lee/pageOne/", "For Lee/pageTwo/", "For Lee/pageThree/"]  # List of directories
TRAIN_CSV = "train.csv"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001  # Lowered learning rate

# Custom Dataset class
class ZebrafishDataset(Dataset):
    def __init__(self, csv_file=None, image_dirs=None, transform=None, data=None):
        if csv_file:
            self.data = pd.read_csv(csv_file)
            # Only keep 'healthy' and 'sb'
            self.data = self.data[self.data['disease'].isin(['healthy', 'sb'])].reset_index(drop=True)
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
        # Find the image in one of the directories
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
        
        # Update __getitem__ to return None for invalid samples
        if image is None or label is None:
            return None

        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Modify the model to include dropout layers
class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedResNet18, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Add dropout layer
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

if __name__ == "__main__":
    # Load dataset (only 'healthy' and 'sb')
    dataset = ZebrafishDataset(csv_file=TRAIN_CSV, image_dirs=IMAGE_DIRS, transform=transform)
    # Split data into training and validation sets
    train_data, val_data = train_test_split(dataset.data, test_size=0.2, random_state=42)
    train_dataset = ZebrafishDataset(image_dirs=IMAGE_DIRS, transform=transform, data=train_data.reset_index(drop=True))
    val_dataset = ZebrafishDataset(image_dirs=IMAGE_DIRS, transform=transform, data=val_data.reset_index(drop=True))

    # Update DataLoader to filter out None values
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        return default_collate(batch)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Use a binary classifier (2 output classes)
    model = ModifiedResNet18(num_classes=2)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # Added weight decay

    # Early stopping parameters
    patience = 5  # Increased patience
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop with early stopping
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            # Convert labels to tensor indices
            # Map 'healthy' to 0, 'sb' to 1
            labels = torch.tensor([0 if label == 'healthy' else 1 for label in labels])
            
            # Move data to device (GPU if available)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            images, labels = images.to(device), labels.to(device)
            model.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                # Convert labels to tensor indices
                labels = torch.tensor([0 if label == 'healthy' else 1 for label in labels])
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss}")

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")  # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")

    # Load the trained model
    model = ModifiedResNet18(num_classes=2)  # Match the saved model
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    # Prepare the test dataset
    test_dataset = ZebrafishDataset(csv_file="test.csv", image_dirs=IMAGE_DIRS, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Evaluate the model on the test dataset
    def evaluate_model(model, test_loader):
        correct = 0
        total = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        with torch.no_grad():
            for images, labels in test_loader:
                # Convert labels to tensor indices
                labels = torch.tensor([0 if label == 'healthy' else 1 for label in labels])
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")

    # Call the evaluation function
    evaluate_model(model, test_loader)