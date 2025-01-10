from torch import clamp, nn

from src.networks.simple.constants import SIMPLE_NETWORK_INPUT_SIZE
from src.model.nnue import NNUE


class SimpleNetwork(NNUE):
    def __init__(self, hidden_size):
        super(SimpleNetwork, self).__init__()
        self.fc1 = nn.Linear(SIMPLE_NETWORK_INPUT_SIZE, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, 1, bias=False)

    def save_weights(self, epoch: int, train_directory: str):
        super().save_weights(epoch, train_directory)
        self._save_weight(self.fc1, "fc1", epoch, train_directory)
        self._save_weight(self.fc2, "fc2", epoch, train_directory)

    def forward(self, x):
        x = clamp(self.fc1(x), 0, 1)
        x = self.fc2(x)
        return x
