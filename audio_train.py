import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from urbansounddataset import UrbanSoundDataset
from cnn_network import CNNNetwork
from audio_inference import run_test
import time


def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_fn, 
          optimiser, device, epochs, epoch, fold):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("-------------")
        epoch += 1
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), 
        },
        "checkpoint.pt"
        )
        if i > 0 and i % 10 == 0:
            run_test(fold, device)

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                prog = 'play',
                description = 'plays two othello players',
                epilog = 'Text at the bottom of help')

    parser.add_argument("-f", "--fold", default=9)
    parser.add_argument("-e", "--epochs", default=50)
    args = parser.parse_args()
    fold = int(args.fold)
    n_epochs = int(args.epochs)

    device = "cpu" # as always with M1 cpu is faster than mps
    # if torch.cuda.is_available():
    #     device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    BATCH_SIZE=128
    EPOCHS = 10
    LEARNING_RATE=0.001

    ANNOTATIONS_FILE = "/Users/bill/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "/Users/bill/UrbanSound8K/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, 
                            AUDIO_DIR, 
                            mel_spectrogram, 
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device,
                            train=True,
                            train_fold=fold)
    print(f"len of dataset {len(usd)}")

    train_data_loader = DataLoader(usd, batch_size=BATCH_SIZE)

    print(f"using {device}")
    # 3. build model
    model = CNNNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if os.path.exists("checkpoint.pt"):
        checkpoint = torch.load("checkpoint.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
    model.train()

    # 4. train
    # time the train run
    start_time = time.time()
    train(model, train_data_loader, loss_fn, 
          optimizer, device, n_epochs, epoch, fold)
    end_time = time.time()
    print(f"Training time: {end_time - start_time}")

    # 5. save trained model
    print("Model trained and stored")
