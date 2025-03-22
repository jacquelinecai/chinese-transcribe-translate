import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

batch_size = 64  

class HandwrittenChineseCNN(nn.Module):
    def __init__(self, num_classes):
        super(HandwrittenChineseCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def load_dataset(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),  
        transforms.RandomRotation(10),  
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)  

    return dataset, dataloader

def train_cnn(model, train_loader, num_epochs, learning_rate):
    if torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using Device:", device)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print("Starting Training...", flush=True)
    for epoch in range(num_epochs):
        print("Epoch " + str(epoch + 1) + "/" + str(num_epochs) + " - LR: " + str(round(float(optimizer.param_groups[0]['lr']), 6)), flush=True)
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        print("Epoch " + str(epoch+1) + "/" + str(num_epochs) + " - Loss: " + str(round(running_loss / len(train_loader), 4)))

    print("Training complete.")
    torch.save(model.state_dict(), "chinese_character_cnn2.pth")

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_directory, "..", "data", "chinese_characters_train")
    num_epochs = 40
    learning_rate = 0.0005

    dataset, train_loader = load_dataset(data_dir, batch_size)

    num_classes = len(dataset.classes)
    print("Number of Classes (Training): " + str(num_classes)) 

    model = HandwrittenChineseCNN(num_classes)
    train_cnn(model, train_loader, num_epochs, learning_rate)
