import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os


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

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path).convert("L")
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def predict_character(image_path, model, class_labels, top_n=5):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    image = preprocess_image(image_path).to(device)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probs, top_n, dim=1)
    
    top_1_char = class_labels[top_indices[0][0].item()]
    # top_2_char = class_labels[top_indices[0][1].item()]

    return top_1_char

def get_folders_from_target_directory(target_directory):
    if not os.path.isdir(target_directory):
        print(f"Directory '{target_directory}' does not exist.")
        return []

    return [entry.name for entry in os.scandir(target_directory) if entry.is_dir()]

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_directory, "..", "data", "chinese_characters_test")
    model_path = os.path.join(current_directory, "chinese_character_cnn4_with_spaces.pth")
    
    class_labels = get_folders_from_target_directory(data_dir)
    class_labels.sort()
    
    num_classes = len(class_labels)
    model = HandwrittenChineseCNN(num_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model_weights = torch.load(model_path, map_location=device)
    model.load_state_dict(model_weights)
    model.eval()
    
    total_images = 0
    correct_predictions = 0
    
    folders = get_folders_from_target_directory(data_dir)
    
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if os.path.isfile(image_path):
                prediction = predict_character(image_path, model, class_labels, top_n=2)
                total_images += 1
                # print('predicted: ' + prediction + ', actual: '+ str(folder))
                if folder == prediction:
                    correct_predictions += 1
    
    accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
    print(f"Accuracy: {accuracy:.2f}%")