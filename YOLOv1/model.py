import torch
import torch.nn as nn

"""
Only has conv layers (without FCNNs)
"""
architecture_config = [
    (7, 64, 2, 3),  # (kernel_size, output_filter_numbers, stride, padding)
    "MaxPool",
    (3, 192, 1, 1),
    "MaxPool",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "MaxPool",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],  # [..., ..., repeat_times]
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "MaxPool",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              bias=False,  # Since we use BatchNorm (not mentioned in original YOLOv1)
                              **kwargs
                              )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return x


class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(YOLOv1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fully_connected = self._create_fully_connected(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        return self.fully_connected(x)

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if isinstance(x, tuple):
                layers += [CNNBlock(in_channels=in_channels,
                                    out_channels=x[1],
                                    kernel_size=x[0],
                                    stride=x[2],
                                    padding=x[3])]

                in_channels = x[1]
            elif isinstance(x, str):
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif isinstance(x, list):
                conv1, conv2, repeat_times = x

                for _ in range(repeat_times):
                    layers += [
                        CNNBlock(in_channels=in_channels,
                                 out_channels=conv1[1],
                                 kernel_size=conv1[0],
                                 stride=conv1[2],
                                 padding=conv1[3]
                                 ),
                        CNNBlock(in_channels=conv1[1],
                                 out_channels=conv2[1],
                                 kernel_size=conv2[0],
                                 stride=conv2[2],
                                 padding=conv2[3]
                                 ),
                    ]

                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    @staticmethod
    def _create_fully_connected(split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),  # In the original paper 496 is 4096 (which takes a lot VRAM)
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5))  # * 5: each bounding box has (x1, x2, y1, y2, confidence_score).
        )


def test(split_size=7, num_boxes=2, num_classes=20):
    model = YOLOv1(in_channels=3, split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)
    x = torch.randn(2, 3, 448, 448)
    print(model(x).shape)

test()
