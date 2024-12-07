import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os

# ==============================================================
# 1. Neural Network Architecture
# ==============================================================
class LicensePlateYOLO(nn.Module):
    """
    Simplified YOLO-like architecture for license plate detection.
    """
    def __init__(self, num_classes=1, grid_size=7, bbox_per_grid=2):
        super(LicensePlateYOLO, self).__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.bbox_per_grid = bbox_per_grid

        # Feature extraction layers (convolutional + pooling)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # RGB input
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Downsampling
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Fully connected layers for grid-based prediction
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            #nn.Linear(128 * (grid_size // 8) ** 2, 1024),  # Reduced size due to pooling
            nn.Linear(100352, 1024),
            nn.ReLU(),
            nn.Linear(1024, grid_size * grid_size * (bbox_per_grid * 5 + num_classes))
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        # Reshape output to (batch_size, grid_size, grid_size, bbox * 5 + num_classes)
        x = x.view(-1, self.grid_size, self.grid_size, self.bbox_per_grid * 5 + self.num_classes)
        return x

# ==============================================================
# 2. Loss Function
# ==============================================================
class YoloLoss(nn.Module):
    """
    Custom loss function combining coordinate loss, confidence loss, and classification loss.
    """
    def __init__(self, lambda_coord=5, lambda_noobj=0.5):
        super(YoloLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        # Separate components from predictions
        pred_bboxes = predictions[..., :4]  # Bounding box: x, y, w, h
        pred_conf = predictions[..., 4:5]  # Confidence score
        pred_class = predictions[..., 5:]  # Class probabilities

        # Separate components from targets
        target_bboxes = targets[..., :4]
        target_conf = targets[..., 4:5]
        target_class = targets[..., 5:]
        #target_class = target_class.view(target_class.size(0), -1)

        # Coordinate loss
        bbox_loss = F.mse_loss(pred_bboxes, target_bboxes)

        # Confidence loss
        conf_loss = F.mse_loss(pred_conf, target_conf)

        # Classification loss
        pred_class = pred_class.reshape(pred_class.size(0), 1, -1)
        target_class = target_class.reshape(target_class.size(0), -1) 
        class_loss = F.cross_entropy(pred_class, target_class, ignore_index=0)

        # Total loss with balancing factors
        total_loss = (
            self.lambda_coord * bbox_loss +
            conf_loss +
            self.lambda_noobj * (1 - target_conf) * conf_loss
        )
        return total_loss

# ==============================================================
# 3. Dataset Preparation
# ==============================================================
class LicensePlateDataset(Dataset):
    """
    Custom Dataset class for loading images and annotations.
    """
    def __init__(self, image_dir, annotation_dir, grid_size=7, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.grid_size = grid_size
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        annotation_path = os.path.join(self.annotation_dir, self.images[idx].replace('.jpg', '.txt'))

        # Load image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))  # Resize to match input size
        image = image.transpose(2, 0, 1) / 255.0  # Normalize and convert to channels-first
        image = torch.tensor(image, dtype=torch.float32)

        # Load annotations
        annotations = np.loadtxt(annotation_path).reshape(-1, 5)  # [class, x, y, w, h]
        target = torch.zeros((self.grid_size, self.grid_size, 6))  # Default target (no objects)

        # Map annotations to grid
        for annotation in annotations:
            class_id, x, y, w, h = annotation
            grid_x = int(x * self.grid_size)
            grid_y = int(y * self.grid_size)
            target[grid_y, grid_x, 0] = x  # Center x
            target[grid_y, grid_x, 1] = y  # Center y
            target[grid_y, grid_x, 2] = w  # Width
            target[grid_y, grid_x, 3] = h  # Height
            target[grid_y, grid_x, 4] = 1.0  # Confidence
            target[grid_y, grid_x, 5] = int(class_id)

        return image, target

# ==============================================================
# 4. Training Loop
# ==============================================================
def train_model(model, dataloader, optimizer, loss_fn, num_epochs=10, device='cuda'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)

            # Forward pass
            predictions = model(images)
            loss = loss_fn(predictions, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

# ==============================================================
# 5. Prediction
# ==============================================================
def predict(model, image_path, grid_size=7, conf_threshold=0.5):
    model.eval()
    with torch.no_grad():
        # Load and preprocess the image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224)).transpose(2, 0, 1) / 255.0
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        # Get predictions
        predictions = model(image)
        predictions = predictions.squeeze(0)

        # Parse predictions
        for row in range(grid_size):
            for col in range(grid_size):
                cell = predictions[row, col]
                confidence = cell[4]

                if confidence > conf_threshold:
                    x, y, w, h = cell[:4]
                    class_id = torch.argmax(cell[5:])
                    print(f"Detected: Class {class_id}, BBox: ({x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}), Confidence: {confidence:.2f}")

# ==============================================================
# 6. Running the Code
# ==============================================================
if __name__ == "__main__":
    # Paths for dataset
    train_image_dir = "../trainData/data/images/train"
    train_annotation_dir = "../trainData/data/labels/train"

    # Hyperparameters
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 10

    # Dataset and DataLoader
    dataset = LicensePlateDataset(train_image_dir, train_annotation_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, and Optimizer
    model = LicensePlateYOLO()
    loss_fn = YoloLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    train_model(model, dataloader, optimizer, loss_fn, num_epochs)