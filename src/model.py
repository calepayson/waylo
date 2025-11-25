import torch
import torch.nn as nn

from config import CurrentConfig as Conf


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolo(nn.Module):
    def __init__(self, config, in_channels=3):
        super(Yolo, self).__init__()
        self.config = config
        self.in_channels = in_channels
        self.conv_layers = self._create_conv_layers(self.config.ARCHITECTURE)
        self.fc_layers = self._create_fc_layers()

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for layer in architecture:
            if type(layer) is tuple:
                layers += [
                    CNNBlock(
                        in_channels,
                        layer[1],
                        kernel_size=layer[0],
                        stride=layer[2],
                        padding=layer[3],
                    )
                ]
                in_channels = layer[1]
            elif type(layer) is str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(layer) is list:
                conv1 = layer[0]
                conv2 = layer[1]
                n_repeats = layer[2]

                for _ in range(n_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]

                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fc_layers(self):
        S = self.config.split_size
        B = self.config.n_boxes
        C = self.config.n_classes
        fc_in = self.config.final_conv_channels * S * S
        fc_hidden = self.config.fc_hidden_size
        fc_out = S * S * (C + B * 5)

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, fc_hidden),
            nn.Dropout(self.config.dropout),
            nn.LeakyReLU(self.config.leaky_relu),
            nn.Linear(fc_hidden, fc_out),
        )


def quick_test(S=7, B=2, C=20):
    config = Conf()
    model = Yolo(config)
    x = torch.randn((2, 3, config.img_size, config.img_size))
    print(model(x).shape)


if __name__ == "__main__":
    quick_test()
