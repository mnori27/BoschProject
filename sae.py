from torch import nn


class SAE(nn.Module):

    def __init__(self):
        """
        háló létrehozása
        külön encoder decoder
        közéjük ReLU..

        """
        super(SAE).__init__()

    def forward(self, x):
        """
        algorithm 1
        encoderen, mdsen, decoderen átküldeni

        """

    def one_step(self):
        """
        forwardon át, loss kiszámol, vissza
        """

    def train(self):
        """
        adat betöltése, forward hívogatása
        """


class Encoder(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size):
        """

        """
        super(Encoder).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=kernel_size)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=kernel_size),
                                   nn.ReLU())
        """
        nn.ConvTranspose2d decoderhez
        """
        self.loss = nn.MSELoss(-1)
