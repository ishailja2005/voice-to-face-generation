import torch 
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(

            nn.Linear(256, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512 * 5 * 5),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()

        )

    
    def forward(self, x):
        x=  self.fc(x)
        x=x.view(-1, 512, 5, 5)
        x=self.deconv(x)
        return x