import torch
from train import FeedForwardNet, download_mnist_datasets

class_mapping = [str(i) for i in range(10)]

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
        return predicted, expected


if __name__ == "__main__":
    # load model built in train
    
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("feedforwardnet.pth")
    feed_forward_net.load_state_dict(state_dict)

    # get a sample from the validation dataset
    # make an inference with the trained model
    # print the predicted digit

    # download validation dataset
    _, val_data = download_mnist_datasets()
    n_good = 0
    n_total = 0
    missed = {}
    for i, (images, labels) in enumerate(val_data):
        sample_image = images[0]
        sample_label = labels

        # make inference
        output = feed_forward_net(sample_image.unsqueeze(0))
        predicted_label = torch.argmax(output.data).item()

        # print predicted digit
        if predicted_label == sample_label:
            n_good += 1
        else:
            missed[(sample_label, predicted_label)] = missed.get((sample_label, predicted_label), 0) + 1
        n_total += 1
    print(f"{n_good} correct predictions out of {n_total}")
    
    # print the list of missed items, sorted by count of misses
    sorted_missed = sorted(missed.items(), key=lambda item: item[1], reverse=True)
    for (sample, predicted), count in sorted_missed:
        print(f"misses: {count} actual: {sample} predicted {predicted}")






