# import libraries
import torch
import torch.nn as nn


def createLevel3FullyConvNet():

    class Level3FullyConvNet(nn.Module):
        def __init__(self, *args, **kwargs):
            super(Level3FullyConvNet, self).__init__(*args, **kwargs)
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 15, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(15),
                # Output size: (15, 7, 7)
                nn.Conv2d(15, 84, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(84),
                # Output size: (84, 5, 5)
                nn.Conv2d(84, 167, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(167),
                # Output size: (167, 3, 3)
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(167, 84, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(84),
                # Output size: (84, 5, 5)
                nn.ConvTranspose2d(84, 15, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(15),
                # Output size: (15, 7, 7)
                nn.ConvTranspose2d(15, 1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                # Output size: (1, 7, 7)
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    # Instantiate the network
    net = Level3FullyConvNet()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-2)

    return net, criterion, optimizer


# def createLevel3FullyConvDropoutNet():

#     class Level3FullyConvDropoutNet(nn.Module):
#         def __init__(self, *args, **kwargs):
#             super(Level3FullyConvDropoutNet, self).__init__(*args, **kwargs)
#             # Encoder
#             self.encoder = nn.Sequential(
#                 nn.Conv2d(1, 15, kernel_size=3, stride=1, padding=1),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(15),
#                 nn.Dropout2d(p=0.2),
#                 # Output size: (15, 7, 7)
#                 nn.Conv2d(15, 84, kernel_size=3, stride=1, padding=0),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(84),
#                 nn.Dropout2d(p=0.2),
#                 # Output size: (84, 5, 5)
#                 nn.Conv2d(84, 167, kernel_size=3, stride=1, padding=0),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(167),
#                 nn.Dropout2d(p=0.2),
#                 # Output size: (167, 3, 3)
#             )

#             # Decoder
#             self.decoder = nn.Sequential(
#                 nn.ConvTranspose2d(167, 84, kernel_size=3, stride=1, padding=0),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(84),
#                 # Output size: (84, 5, 5)
#                 nn.ConvTranspose2d(84, 15, kernel_size=3, stride=1, padding=0),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(15),
#                 # Output size: (15, 7, 7)
#                 nn.ConvTranspose2d(15, 1, kernel_size=3, stride=1, padding=1),
#                 nn.ReLU(),
#                 # Output size: (1, 7, 7)
#             )

#         def forward(self, x):
#             x = self.encoder(x)
#             x = self.decoder(x)
#             return x

#     # Instantiate the network
#     net = Level3FullyConvDropoutNet()

#     # Loss function and optimizer
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-2)

#     return net, criterion, optimizer


def createLevel3FullyConvDropoutNet(channels):

    class Level3FullyConvDropoutNet(nn.Module):
        def __init__(self):
            super(Level3FullyConvDropoutNet, self).__init__()

            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                nn.Dropout2d(p=0.2),

                nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                nn.Dropout2d(p=0.2),

                nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
            )
            
            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(channels[2], channels[1], kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),

                nn.ConvTranspose2d(channels[1], channels[0], kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),

                nn.ConvTranspose2d(channels[0], 1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )

        def forward(self, x: torch.Tensor):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    # Instantiate the network
    net = Level3FullyConvDropoutNet()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-2)

    return net, criterion, optimizer
