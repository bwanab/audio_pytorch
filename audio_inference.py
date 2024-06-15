import torch
import torchaudio
from urbansounddataset import UrbanSoundDataset, ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES
from cnn_network import CNNNetwork

class_mapping = [str(i) for i in range(10)]

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
        return predicted, expected

import argparse
def run_test(fold, device):
    cnn = CNNNetwork()
    checkpoint = torch.load("checkpoint.pt")
    cnn.load_state_dict(checkpoint["model_state_dict"])
    cnn.eval()

    # get a sample from the validation dataset
    # make an inference with the trained model
    # print the predicted digit
    
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
                            train=False,
                            train_fold=fold)

    # download urban sound dataset
    
    n_good = 0
    n_total = 0
    missed = {}
    # input, target = usd[0][0], usd[0][1]
    # input.unsqueeze_(0)
    # output = cnn(input)
    # predicted_label = torch.argmax(output.data).item()
    # print(f"Predicted: {predicted_label} expected: {target}")

    for i, (audios, labels) in enumerate(usd):
        sample_audio = audios
        sample_label = labels
        sample_audio.unsqueeze_(0)
        # make inference
        output = cnn(sample_audio)
        predicted_label = torch.argmax(output.data).item()

        # print predicted digit
        if predicted_label == sample_label:
            n_good += 1
        else:
            missed[(sample_label, predicted_label)] = missed.get((sample_label, predicted_label), 0) + 1
        n_total += 1
    print(f"fold{fold}: {n_good} correct predictions out of {n_total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                prog = 'play',
                description = 'plays two othello players',
                epilog = 'Text at the bottom of help')

    parser.add_argument("-f", "--fold", default=9)
    args = parser.parse_args()
    fold = int(args.fold)

    device = "cpu" # as always with M1 cpu is faster than mps
    # if torch.cuda.is_available():
    #     device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = "mps"

    # load model built in train
    
    run_test(fold, device)
    
    # # print the list of missed items, sorted by count of misses
    # sorted_missed = sorted(missed.items(), key=lambda item: item[1], reverse=True)
    # for (sample, predicted), count in sorted_missed:
    #     print(f"misses: {count} actual: {sample} predicted {predicted}")






