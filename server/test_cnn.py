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

        for i in range(top_n):
            predicted_char = class_labels[top_indices[0][i].item()]
            probability = top_probs[0][i].item() * 100
            print("Dear User, I am " + str(round(probability, 3)) + " % sure it's " + predicted_char)

    return class_labels[top_indices[0][0].item()]

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_directory, "..", "data", "chinese_characters_test")
    model_path = os.path.join(current_directory, "chinese_character_cnn.pth")

    class_labels = []
    for d in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, d)):
            class_labels.append(d)

    class_labels.sort()

    num_classes = len(class_labels)
    model = HandwrittenChineseCNN(num_classes)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    model_weights = torch.load(model_path, map_location=device)
    model.load_state_dict(model_weights)
    model.eval()

    try:
        top_n = int(input("\nPlease specify number of predictions per input image:\n"))
    except ValueError:
        top_n = 5  #default

    while True:
        user_input_path = input("\nEnter the path of the test image:\n").strip()

        if not os.path.isabs(user_input_path):
            test_image_path = os.path.abspath(os.path.join(current_directory, "..", user_input_path))
        else:
            test_image_path = user_input_path

        print("Full path: " + str(repr(test_image_path)))
        if not os.path.exists(test_image_path):
            print("Error: Image path " + str(test_image_path) + " not found! Try again.")
            continue

        predicted_char = predict_character(test_image_path, model, class_labels, top_n)

        cont = input("\nContinue? [Y/n] ").strip().lower()
        if cont == "n":
            break
