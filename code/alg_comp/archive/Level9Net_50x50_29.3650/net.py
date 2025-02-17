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
                nn.ConvTranspose2d(channels[7], channels[6], kernel_size=2, stride=2, padding=0),
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
    net = Level9Net()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-2)

    return net, criterion, optimizer