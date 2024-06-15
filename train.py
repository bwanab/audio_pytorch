import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time

# 1. download dataset
# 2. create data loader
# 3. build model
# 4. train
# 5. save trained model

class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions

BATCH_SIZE=128
EPOCHS = 10
LEARNING_RATE=0.001

def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor()
    )
    validation_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )
    return train_data, validation_data

train_data, validation_data = download_mnist_datasets()

def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    print(f"Loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("-------------")
    print("Training done")

if __name__ == "__main__":
    device = "cpu" # as always with M1 cpu is faster than mps
    # if torch.cuda.is_available():
    #     device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = "mps"

    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    print(f"using {device}")
    # 1. download dataset
    # 2. create data loader
    train_data, _ = download_mnist_datasets()

    # 3. build model
    model = FeedForwardNet().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 4. train
    # time the train run
    start_time = time.time()
    train(model, train_data_loader, loss_fn, optimizer, device, EPOCHS)
    end_time = time.time()
    print(f"Training time: {end_time - start_time}")

    # 5. save trained model
    torch.save(model.state_dict(), "feedforwardnet.pth")
    print("Model trained and stored")
