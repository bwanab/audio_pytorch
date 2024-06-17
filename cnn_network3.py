from torch import nn
from torchsummary import summary

class CNNNetwork3(nn.Module):
    def __init__(self, channel_defs=[(1, 16), (16, 32), (32, 64), (64, 128), (128, 128), (128, 128), (128, 64)]):
        super().__init__()
        def conv_layer(in_c, out_c):
            conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=3,
                    stride=1,
                    padding=2
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            return conv
        
        self.net = nn.ModuleList([conv_layer(in_c, out_c) for in_c, out_c in channel_defs])
        self.net.append(nn.Flatten())
        self.net.append(nn.Linear(256, 10))

    def forward(self, input_data):
        x = input_data
        for layer in self.net:
            x = layer(x)
        return x
    

if __name__ == "__main__":
    cnn = CNNNetwork3()
    summary(cnn, (1, 64, 44))
