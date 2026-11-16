import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model
model = Net().to(device)
model.load_state_dict(torch.load(
    r"D:\MyPersonalWork\mathsMachineLearning\models\mnist_model.pth",
    map_location=device
))
model.eval()

# Transform for real-world image
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# Change this to your image file
img_path = r"D:\MyPersonalWork\mathsMachineLearning\img_7639.jpg"

img = Image.open(img_path)

img = transform(img)
img = img.unsqueeze(0)
img = img.to(device)

# Prediction
with torch.no_grad():
    output = model(img)
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

print("Predicted Digit:", predicted.item())
print("Confidence:", confidence.item())