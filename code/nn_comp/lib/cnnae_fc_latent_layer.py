# Import libraries
import torch
import torch.nn as nn


def createLevel3FullyConnectedNet():

    class Level3FullyConnectedNet(nn.Module):
        def __init__(self, *args, **kwargs):
            super(Level3FullyConnectedNet, self).__init__(*args, **kwargs)

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

            # Latent fully connected layer
            self.fc_latent = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    167 * 3 * 3, 256
                ),  # Compress to latent space (e.g., 256 units)
                nn.ReLU(),
                nn.Linear(256, 167 * 3 * 3),  # Re-expand from latent space
                nn.ReLU(),
                nn.Unflatten(1, (167, 3, 3)),
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
            x = self.fc_latent(x)  # Pass through the fully connected latent layer
            x = self.decoder(x)
            return x

    # Instantiate the network
    net = Level3FullyConnectedNet()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-2)

    return net, criterion, optimizer
