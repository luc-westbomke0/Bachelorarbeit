import torch
import torch.nn as nn


def createLevel5Net(channels=((15, 37, 84, 122, 167))):

    class Level5Net(nn.Module):
        def __init__(self):
            super(Level5Net, self).__init__()
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, channels[0], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                nn.Dropout2d(p=0.2),
                nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                nn.Dropout2d(p=0.2),
                nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
                nn.Dropout2d(p=0.2),
                nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[3]),
                nn.Dropout2d(p=0.2),
                nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[4]),
                # nn.Dropout2d(p=0.2),
            )
            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(
                    channels[4], channels[3], kernel_size=3, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.BatchNorm2d(channels[3]),
                nn.ConvTranspose2d(
                    channels[3], channels[2], kernel_size=3, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
                nn.ConvTranspose2d(
                    channels[2], channels[1], kernel_size=3, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                nn.ConvTranspose2d(
                    channels[1], channels[0], kernel_size=3, stride=2, padding=0
                ),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                nn.ConvTranspose2d(channels[0], 1, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )

        def forward(self, x: torch.Tensor):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    # Instantiate the network
    net = Level5Net()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-2)

    return net, criterion, optimizer


def createLevel7Net30x30(channels=(32, 32, 64, 128, 128, 256, 256), print_shape=False):
    class Level7Net(nn.Module):
        def __init__(self):
            super(Level7Net, self).__init__()

            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(channels[0], channels[1], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
                nn.Conv2d(channels[2], channels[3], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[3]),
                nn.Conv2d(channels[3], channels[4], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[4]),
                nn.Conv2d(channels[4], channels[5], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[5]),
                nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[6]),
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(channels[6], channels[5], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[5]),
                nn.ConvTranspose2d(channels[5], channels[4], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[4]),
                nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2, padding=0, output_padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[3]),
                nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2, padding=0, output_padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
                nn.ConvTranspose2d(channels[2], channels[1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                nn.ConvTranspose2d(channels[0], 1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )

        def forward(self, x: torch.Tensor):
            if print_shape: print(f"Input shape: {x.shape}")
            x = self.encoder(x)
            if print_shape: print(f"Latent shape: {x.shape}")
            x = self.decoder(x)
            if print_shape: print(f"Output shape: {x.shape}")
            return x

    # Instantiate the network
    net = Level7Net()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-2)

    return net, criterion, optimizer


def createLevel9Net(channels=(32, 32, 64, 64, 128, 128, 256, 256, 512), print_shape=False):
    class Level9Net(nn.Module):
        def __init__(self):
            super(Level9Net, self).__init__()

            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(channels[0], channels[1], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
                nn.Conv2d(channels[2], channels[3], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[3]),
                nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[4]),
                nn.Conv2d(channels[4], channels[5], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[5]),
                nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[6]),
                nn.Conv2d(channels[6], channels[7], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[7]),
                nn.Conv2d(channels[7], channels[8], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[8]),
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(channels[8], channels[7], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[7]),
                nn.ConvTranspose2d(channels[7], channels[6], kernel_size=2, stride=2, padding=0, output_padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[6]),
                nn.ConvTranspose2d(channels[6], channels[5], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[5]),
                nn.ConvTranspose2d(channels[5], channels[4], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[4]),
                nn.ConvTranspose2d(channels[4], channels[3], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[3]),
                nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
                nn.ConvTranspose2d(channels[2], channels[1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                nn.ConvTranspose2d(channels[0], 1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )

        def forward(self, x: torch.Tensor):
            if print_shape: print(f"Input shape: {x.shape}")
            x = self.encoder(x)
            # if print_shape: print(f"Latent shape input: {x.shape}")
            # x = self.latent(x)
            if print_shape: print(f"Latent shape: {x.shape}")
            x = self.decoder(x)
            if print_shape: print(f"Output shape: {x.shape}")
            return x

    # Instantiate the network
    net = Level9Net()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-2)

    return net, criterion, optimizer

def createLevel8Net(channels=(32, 32, 64, 64, 128, 128, 256, 256, 512), print_shape=False):
    class Level8Net(nn.Module):
        def __init__(self):
            super(Level8Net, self).__init__()

            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(channels[0], channels[1], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
                nn.Conv2d(channels[2], channels[3], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[3]),
                nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[4]),
                nn.Conv2d(channels[4], channels[5], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[5]),
                nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                # nn.BatchNorm2d(channels[6]),
                # nn.Conv2d(channels[6], channels[7], kernel_size=2, stride=2, padding=0),
                # nn.ReLU(),
                nn.BatchNorm2d(channels[7]),
                nn.Conv2d(channels[7], channels[8], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[8]),
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(channels[8], channels[7], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                # nn.BatchNorm2d(channels[7]),
                # nn.ConvTranspose2d(channels[7], channels[6], kernel_size=2, stride=2, padding=0, output_padding=1),
                # nn.ReLU(),
                nn.BatchNorm2d(channels[6]),
                nn.ConvTranspose2d(channels[6], channels[5], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[5]),
                nn.ConvTranspose2d(channels[5], channels[4], kernel_size=2, stride=2, padding=0, output_padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[4]),
                nn.ConvTranspose2d(channels[4], channels[3], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[3]),
                nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2, padding=0, output_padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
                nn.ConvTranspose2d(channels[2], channels[1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                nn.ConvTranspose2d(channels[0], 1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )

        def forward(self, x: torch.Tensor):
            if print_shape: print(f"Input shape: {x.shape}")
            x = self.encoder(x)
            # if print_shape: print(f"Latent shape input: {x.shape}")
            # x = self.latent(x)
            if print_shape: print(f"Latent shape: {x.shape}")
            x = self.decoder(x)
            if print_shape: print(f"Output shape: {x.shape}")
            return x

    # Instantiate the network
    net = Level8Net()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-2)

    return net, criterion, optimizer

def createLevel6Net(channels=(32, 32, 64, 128, 128, 256, 256), print_shape=False):
    class Level6Net(nn.Module):
        def __init__(self):
            super(Level6Net, self).__init__()

            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(channels[0], channels[1], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
                nn.Conv2d(channels[2], channels[3], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                # nn.BatchNorm2d(channels[3]),
                # nn.Conv2d(channels[3], channels[4], kernel_size=2, stride=2, padding=0),
                # nn.ReLU(),
                nn.BatchNorm2d(channels[4]),
                nn.Conv2d(channels[4], channels[5], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[5]),
                nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[6]),
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(channels[6], channels[5], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[5]),
                nn.ConvTranspose2d(channels[5], channels[4], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                # nn.BatchNorm2d(channels[4]),
                # nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2, padding=0),
                # nn.ReLU(),
                nn.BatchNorm2d(channels[3]),
                nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2, padding=0, output_padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
                nn.ConvTranspose2d(channels[2], channels[1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                nn.ConvTranspose2d(channels[0], 1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )

        def forward(self, x: torch.Tensor):
            if print_shape: print(f"Input shape: {x.shape}")
            x = self.encoder(x)
            if print_shape: print(f"Latent shape: {x.shape}")
            x = self.decoder(x)
            if print_shape: print(f"Output shape: {x.shape}")
            return x

    # Instantiate the network
    net = Level6Net()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-2)

    return net, criterion, optimizer


def createLevel6Net_10x10(channels=(32, 32, 64, 128, 256, 256), print_shape=False):
    class Level6Net(nn.Module):
        def __init__(self):
            super(Level6Net, self).__init__()

            # input 1@10x10

            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                nn.Dropout2d(p=0.1),
                # 32@10x10

                nn.Conv2d(channels[0], channels[1], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                nn.Dropout2d(p=0.1),
                # 32@5x5

                nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
                # 64@5x5

                nn.Conv2d(channels[2], channels[3], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[3]),
                # 128@2x2

                nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[4]),
                # 256@2x2

                nn.Conv2d(channels[4], channels[5], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[5]),
            )

            # 256@2x2 latent

            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(channels[5], channels[4], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[4]),
                # 256@2x2

                nn.ConvTranspose2d(channels[4], channels[3], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[3]),
                # 128@2x2

                nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2, padding=0, output_padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
                # 64@5x5

                nn.ConvTranspose2d(channels[2], channels[1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                # 32@5x5

                nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                # 32@10x10

                nn.ConvTranspose2d(channels[0], 1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                # 1@10x10
            )

        def forward(self, x: torch.Tensor):
            if print_shape: print(f"Input shape: {x.shape}")
            x = self.encoder(x)
            if print_shape: print(f"Latent shape: {x.shape}")
            x = self.decoder(x)
            if print_shape: print(f"Output shape: {x.shape}")
            return x

    # Instantiate the network
    net = Level6Net()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-2)

    return net, criterion, optimizer

def createLevel7Net_20x20(channels=(32, 32, 64, 128, 128, 256, 256), print_shape=False):
    class Level7Net(nn.Module):
        def __init__(self):
            super(Level7Net, self).__init__()

            # Input 1@20x20

            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                nn.Dropout2d(p=0.1),
                # 32@20x20

                nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                nn.Dropout2d(p=0.1),
                # 32@20x20

                nn.Conv2d(channels[1], channels[2], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
                # 64@10x10

                nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[3]),
                # 128@10x10

                nn.Conv2d(channels[3], channels[4], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[4]),
                # 128@5x5

                nn.Conv2d(channels[4], channels[5], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[5]),
                # 256@5x5

                nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[6]),
            )

            # Latent 256@5x5

            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(channels[6], channels[5], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[5]),
                # 256@5x5

                nn.ConvTranspose2d(channels[5], channels[4], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[4]),
                # 128@5x5

                nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[3]),
                # 128@10x10

                nn.ConvTranspose2d(channels[3], channels[2], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
                # 64@10x10

                nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                # 32@20x20

                nn.ConvTranspose2d(channels[1], channels[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                # 32@20x20

                nn.ConvTranspose2d(channels[0], 1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                # Output: 1@20x20
            )

        def forward(self, x: torch.Tensor):
            if print_shape: print(f"Input shape: {x.shape}")
            x = self.encoder(x)
            if print_shape: print(f"Latent shape: {x.shape}")
            x = self.decoder(x)
            if print_shape: print(f"Output shape: {x.shape}")
            return x

    # Instantiate the network
    net = Level7Net()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-2)

    return net, criterion, optimizer

def createLevel8Net_30x30(channels=(32, 32, 64, 64, 128, 128, 256, 512), print_shape=False):
    class Level8Net(nn.Module):
        def __init__(self):
            super(Level8Net, self).__init__()

            # Input 1@30x30
    
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                nn.Dropout2d(p=0.1),
                # 32@30x20

                nn.Conv2d(channels[0], channels[1], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                nn.Dropout2d(p=0.1),
                # 32@15x15

                nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
                # 64@15x15

                nn.Conv2d(channels[2], channels[3], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[3]),
                # 64@7x7

                nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[4]),
                # 128@7x7

                nn.Conv2d(channels[4], channels[5], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[5]),
                # 128@3x3

                nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[6]),
                # 256@3x3

                nn.Conv2d(channels[6], channels[7], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[7]),
            )

            # Latent 512@3x3

            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(channels[7], channels[6], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[6]),
                # 256@3x3

                nn.ConvTranspose2d(channels[6], channels[5], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[5]),
                # 128@3x3

                nn.ConvTranspose2d(channels[5], channels[4], kernel_size=2, stride=2, padding=0, output_padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[4]),
                # 128@7x7

                nn.ConvTranspose2d(channels[4], channels[3], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[3]),
                # 64@7x7

                nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2, padding=0, output_padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
                # 64@15x15

                nn.ConvTranspose2d(channels[2], channels[1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                # 32@15x15

                nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                # 32@30x30

                nn.ConvTranspose2d(channels[0], 1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                # Output: 1@30x30
            )

        def forward(self, x: torch.Tensor):
            if print_shape: print(f"Input shape: {x.shape}")
            x = self.encoder(x)
            # if print_shape: print(f"Latent shape input: {x.shape}")
            # x = self.latent(x)
            if print_shape: print(f"Latent shape: {x.shape}")
            x = self.decoder(x)
            if print_shape: print(f"Output shape: {x.shape}")
            return x

    # Instantiate the network
    net = Level8Net()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-2)

    return net, criterion, optimizer

def createLevel8Net_40x40(channels=(32, 32, 64, 64, 128, 128, 256, 512), print_shape=False):
    class Level8Net(nn.Module):
        def __init__(self):
            super(Level8Net, self).__init__()

            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(channels[0], channels[1], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
                nn.Conv2d(channels[2], channels[3], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[3]),
                nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[4]),
                nn.Conv2d(channels[4], channels[5], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[5]),
                nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[6]),
                nn.Conv2d(channels[6], channels[7], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[7]),
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(channels[7], channels[6], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[6]),
                nn.ConvTranspose2d(channels[6], channels[5], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[5]),
                nn.ConvTranspose2d(channels[5], channels[4], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[4]),
                nn.ConvTranspose2d(channels[4], channels[3], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[3]),
                nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
                nn.ConvTranspose2d(channels[2], channels[1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                nn.ConvTranspose2d(channels[0], 1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )

        def forward(self, x: torch.Tensor):
            if print_shape: print(f"Input shape: {x.shape}")
            x = self.encoder(x)
            # if print_shape: print(f"Latent shape input: {x.shape}")
            # x = self.latent(x)
            if print_shape: print(f"Latent shape: {x.shape}")
            x = self.decoder(x)
            if print_shape: print(f"Output shape: {x.shape}")
            return x

    # Instantiate the network
    net = Level8Net()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-2)

    return net, criterion, optimizer

def createLevel8Net_50x50(channels=(32, 32, 64, 64, 128, 128, 256, 512), print_shape=False):
    class Level8Net(nn.Module):
        def __init__(self):
            super(Level8Net, self).__init__()

            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(channels[0], channels[1], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
                nn.Conv2d(channels[2], channels[3], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[3]),
                nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[4]),
                nn.Conv2d(channels[4], channels[5], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[5]),
                nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[6]),
                nn.Conv2d(channels[6], channels[7], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[7]),
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(channels[7], channels[6], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[6]),
                nn.ConvTranspose2d(channels[6], channels[5], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[5]),
                nn.ConvTranspose2d(channels[5], channels[4], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[4]),
                nn.ConvTranspose2d(channels[4], channels[3], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[3]),
                nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2, padding=0, output_padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[2]),
                nn.ConvTranspose2d(channels[2], channels[1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(channels[1]),
                nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(channels[0]),
                nn.ConvTranspose2d(channels[0], 1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )

        def forward(self, x: torch.Tensor):
            if print_shape: print(f"Input shape: {x.shape}")
            x = self.encoder(x)
            # if print_shape: print(f"Latent shape input: {x.shape}")
            # x = self.latent(x)
            if print_shape: print(f"Latent shape: {x.shape}")
            x = self.decoder(x)
            if print_shape: print(f"Output shape: {x.shape}")
            return x

    # Instantiate the network
    net = Level8Net()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-2)

    return net, criterion, optimizer
