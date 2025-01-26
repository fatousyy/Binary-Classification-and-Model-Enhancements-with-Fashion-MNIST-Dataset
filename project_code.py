import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# Define a custom dataset class to filter for two classes
class TwoClassDataset(torchvision.datasets.FashionMNIST):
    def __init__(self, *args, classes=[0, 1], **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self.data[torch.isin(self.targets, torch.tensor(classes))]
        self.targets = self.targets[torch.isin(self.targets, torch.tensor(classes))]

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input size is 28x28 (Fashion-MNIST)
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, 1)  # Output layer with 1 neuron for binary classification
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = self.fc1(x)
        x = self.relu(x)
        x = self.output(x)  # No sigmoid here, as we're using BCEWithLogitsLoss!!
        return x

# Instantiate the model
model = SimpleNN()

# Define Binary Cross-Entropy loss and SGD optimizer
criterion = nn.BCEWithLogitsLoss()  # For binary classification, use BCE with logits
optimizer = optim.SGD(model.parameters(), lr=0.001)  # Reduced learning rate

# Load the Fashion-MNIST dataset, filtering for classes 0 and 1
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to mean 0.5 and std 0.5
])

train_dataset = TwoClassDataset(root='./data', train=True, download=True, transform=transform, classes=[0, 1])
test_dataset = TwoClassDataset(root='./data', train=False, download=True, transform=transform, classes=[0, 1])

# Split the training data into training and validation (90% training, 10% validation)
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Function to initialize weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

# Initialize the model weights
model.apply(init_weights)

# Updated training function with learning rate adjustment
def train(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            labels = labels.unsqueeze(1).float()  # Ensure labels match the output shape

            # Calculate loss
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagation

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()  # Update weights

            running_loss += loss.item()

        # Calculate validation loss
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                labels = labels.unsqueeze(1).float()  # Ensure labels match the output shape
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # Print the losses
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}")

# Train the model
train(model, train_loader, val_loader, criterion, optimizer, epochs=10)

#  Modified Neural Network with Convolutional Layers
class ModifiedNN(nn.Module):
    def __init__(self):
        super(ModifiedNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)  # Convolutional layer
        self.pool = nn.MaxPool2d(kernel_size=2)  # Max pooling layer
        self.fc1 = nn.Linear(32 * 13 * 13, 128)  # Adjusted input size for fully connected layer
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, 1)  # Output layer for binary classification

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Convolution + ReLU + Max pooling
        x = x.view(-1, 32 * 13 * 13)  # Flatten the output
        x = self.fc1(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Instantiate the modified model
modified_model = ModifiedNN()

# Define Binary Cross-Entropy loss and SGD optimizer for modified model
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(modified_model.parameters(), lr=0.01)  # Learning rate of 0.01

# Train the modified model
train(modified_model, train_loader, val_loader, criterion, optimizer, epochs=10)

#  Fine-tune a Pre-trained ResNet18 Model
from torchvision import models

# Load a pre-trained ResNet18 model
resnet18 = models.resnet18(weights='DEFAULT')  # Updated to use weights parameter

# Modify the first layer to accept single-channel input (Fashion-MNIST)
resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Modify the final layer to output 2 classes for binary classification
num_features = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_features, 1)  # 1 class for binary classification

# Freeze all layers except the final fully connected layer
for param in resnet18.parameters():
    param.requires_grad = False
for param in resnet18.fc.parameters():
    param.requires_grad = True

# Define Binary Cross-Entropy loss and SGD optimizer for ResNet
criterion = nn.BCEWithLogitsLoss()  # Change loss for binary classification
optimizer = optim.SGD(resnet18.parameters(), lr=0.01)  # Learning rate of 0.01

# Train the ResNet18 model
train(resnet18, train_loader, val_loader, criterion, optimizer, epochs=5)

# Generate Class Activation Maps (CAMs)
import numpy as np
import matplotlib.pyplot as plt


#ISSUE HERE Breakpoint --> fixed !
import torch
import torch.nn as nn

# Hook to capture the feature map
feature_map = None

def hook_fn(module, input, output):
    global feature_map
    feature_map = output  # Save the output of the layer

def get_cam(model, input_image, target_class):
    model.eval()
    
    # Register hook
    hook = model.layer4[-1].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        # Get the output from the model
        output = model(input_image)
        
        # Get the weights of the target class
        if model.fc.out_features == 1:
            weights = model.fc.weight.squeeze()  # Remove dimensions
            target_class = 0  # Only one output
        else:
            weights = model.fc.weight[target_class]

        # After the forward pass, the feature_map is filled
        global feature_map
        
        # Reshape the feature map to (512, N) where N is the number of spatial locations
        feature_map = feature_map.view(feature_map.size(1), -1)  # Reshape to (512, N)

        # Compute the Class Activation Map (CAM)
        weights = weights.unsqueeze(0)  # Ensure weights are (1, 512)
        cam = torch.matmul(weights, feature_map)  # Now weights (1, 512) x feature_map (512, N)

        cam = cam.cpu().numpy()  # Convert to numpy for visualization

        # Remove the hook
        hook.remove()
        
        return cam


# Example to generate CAM for a sample from validation set
def plot_cam(image, cam):
    plt.imshow(image.squeeze(), cmap='gray')  # Original image
    plt.imshow(cam, cmap='jet', alpha=0.5)  # Overlay CAM
    plt.axis('off')
    plt.show()

# Get a sample image and its label from the validation set
sample_image, sample_label = val_dataset[0]
sample_image = sample_image.unsqueeze(0)  # Add batch dimension

# Get CAM for the predicted class
cam = get_cam(resnet18, sample_image, sample_label)  # Use sample_label directly
plot_cam(sample_image[0], cam)


# Intersection over Union (IoU) and Non-Maximum Suppression (NMS)

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def non_maximum_suppression(boxes, scores, threshold):
    keep = []
    indices = scores.argsort()[::-1]  # Sort scores in descending order

    while indices.size > 0:
        current = indices[0]
        keep.append(current)
        
        # Compare current box with others
        if indices.size == 1:
            break
        
        ious = np.array([calculate_iou(boxes[current], boxes[i]) for i in indices[1:]])
        indices = indices[np.where(ious < threshold)[0] + 1]  # Keep boxes with IoU less than threshold

    return keep

# Example usage of IoU and NMS
boxes = np.array([[10, 10, 20, 20], [12, 12, 22, 22], [30, 30, 40, 40]])  # Example bounding boxes
scores = np.array([0.9, 0.85, 0.75])  # Example scores
threshold = 0.3  # IoU threshold

# Perform NMS
selected_indices = non_maximum_suppression(boxes, scores, threshold)
print("Selected indices after NMS:", selected_indices)
