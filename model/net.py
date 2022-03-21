from torch import nn


class Net(nn.Module):
    """
    Feed forward network with a four convolution-max-pooling blocks act as a feature extractor
    and the dataset tasks specific classifier
    """
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(22, 25, kernel_size=1, stride=(3, 1)),
            nn.MaxPool2d(1),
            nn.Conv2d(25, 50, kernel_size=1, stride=(3, 1)),
            nn.MaxPool2d(1),
            nn.Conv2d(50, 100, kernel_size=1, stride=(3, 1)),
            nn.MaxPool2d(1),
            nn.Conv2d(100, 200, kernel_size=1, stride=(3, 1)),
            nn.ReLU(),
            nn.Dropout2d(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2800, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(5, 4),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.classifier(features)
        return logits
