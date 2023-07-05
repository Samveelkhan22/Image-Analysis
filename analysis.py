import torch
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

# Define the path to your dataset
dataset_path = r'C:/Users/J.I Traders/Downloads/ImageFolder'

# Define the transformation to be applied to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Create the ImageFolder dataset
train_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)

# Create a data loader for the dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the number of epochs
num_epochs = 5

# Iterate over the specified number of epochs
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    # Iterate over the data loader to access the images and labels
    for images, labels in train_loader:
        # Perform your desired operations on the images and labels
        # Here, you can print the shape of the images and labels as an example
        print('Image shape:', images.shape)
        print('Label shape:', labels.shape)
        
        # Perform your desired visualizations
        # For example, you can plot the first image in the batch
        plt.imshow(images[0].permute(1, 2, 0))  # Convert tensor shape (C, H, W) to (H, W, C)
        plt.title(f"Label: {labels[0]}")
        plt.show()
