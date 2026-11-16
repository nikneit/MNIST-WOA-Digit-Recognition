import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
print("Using device:", device)

# ---------------- PATH SETUP ----------------
model_dir = r"D:\MyPersonalWork\mathsMachineLearning\models"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "mnist_model.pth")# it stores trained model parameters.
acc_file = os.path.join(model_dir, "best_accuracy.txt")# it stores best accuracy of model.

# ---------------- TRANSFORM ----------------
transform = transforms.ToTensor() #matrices of datas.

# ---------------- LOAD DATA ----------------
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True) 
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ---------------- MODEL ----------------
class Net(nn.Module):
    def __init__(self, dropout_rate): #the constructor will be called at generation of new model.
        super(Net, self).__init__() # initializing base class
        self.flatten = nn.Flatten() # converts 2d vector into 1d vector.(hidden input layer)
        self.fc1 = nn.Linear(28*28, 128) #(hidden input layer)
        self.dropout = nn.Dropout(dropout_rate) # it uses randomly on or off neurons(dropout layer to prevent overfitting.)
        self.fc2 = nn.Linear(128, 10) # (output layer)

    def forward(self, x): # it is responsible for data flowing in neural network.
        x = self.flatten(x)
        x = torch.relu(self.fc1(x)) # activation function for hidden layer and make all negative values zero while keeping positive values unchanged.
        x = self.dropout(x) #prevent overfitting 
        x = self.fc2(x) # it is a final prediction score
        return x

# ---------------- FITNESS FUNCTION ----------------
def fitness(dropout):
    model = Net(dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) #adam-smart gradient descent(learning rate(hyperparameter))

    model.train()
    for epoch in range(2):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad() #reseting gradient 
            output = model(data)
            loss = criterion(output, target)
            loss.backward() #compute gradients
            optimizer.step()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(): # does not track gradient computation.
        for data, target in test_loader: 
            data, target = data.to(device), target.to(device)
            output = model(data) #stores confidence scores
            _, predicted = torch.max(output, 1) #select the highest confidence score
            correct += (predicted == target).sum().item()
            total += target.size(0)

    return correct / total # returning accuracy to the woa function

# ---------------- WOA ----------------
def whale_optimization():
    whales = np.random.uniform(0.1, 0.7, 5)
    best_dropout = 0
    best_score = 0

    for iteration in range(5):
        print(f"\nWOA Iteration {iteration+1}")

        for i in range(len(whales)):
            score = fitness(whales[i])
            print(f"Dropout {whales[i]:.3f} → Accuracy {score:.4f}")

            if score > best_score:
                best_score = score
                best_dropout = whales[i]

        whales = whales + np.random.uniform(-0.1, 0.1, 5)
        whales = np.clip(whales, 0.1, 0.7)

    return best_dropout

# ---------------- RUN WOA ----------------
best_dropout = whale_optimization()
print("\nBest Dropout Found:", best_dropout)

# ---------------- FINAL TRAINING ----------------
final_model = Net(best_dropout).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(final_model.parameters(), lr=0.001)

final_model.train()
for epoch in range(3):
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = final_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f"Final Training Epoch {epoch+1} completed")

# ---------------- EVALUATION FUNCTION ----------------
def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(): #temperally disables gradient computation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    return correct / total

# ---------------- EVALUATE ----------------
new_accuracy = evaluate(final_model)
print("\nNew Model Accuracy:", new_accuracy)

# ---------------- LOAD PREVIOUS BEST ----------------
if os.path.exists(acc_file):
    with open(acc_file, "r") as f:
        best_accuracy = float(f.read())
else:
    best_accuracy = 0

print("Previous Best Accuracy:", best_accuracy)

# ---------------- SAVE ONLY IF IMPROVED ----------------
if new_accuracy > best_accuracy:
    torch.save(final_model.state_dict(), model_path)

    with open(acc_file, "w") as f:
        f.write(str(new_accuracy))

    print("✅ Model improved and saved!")
    print("📁 Saved at:", model_path)
else:
    print("❌ Model not improved. Old model kept.")