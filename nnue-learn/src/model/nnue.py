from abc import abstractmethod

import torch
from torch import nn


def save_layer_weights(weights: nn.Linear, filename):
    weight_matrix = weights.weight.cpu().data.numpy()  # shape [out_features, in_features]
    flat_weights = weight_matrix.flatten()  # shape [out_features * in_features]

    with open(filename, 'w') as file:
        file.write(','.join(str(x) for x in flat_weights))
        file.write('\n')


class NNUE(nn.Module):

    def _save_weight(self, layer: nn.Linear, name: str, epoch: int, train_directory: str):
        save_layer_weights(layer, f"{train_directory}/{name}.{epoch}.weights.csv")

    def save_weights(self, epoch: int, train_directory: str):
        model = self.state_dict()
        torch.save(model, f"{train_directory}/model.{epoch}.pth")

    def load_weights(self, epoch: int, train_directory: str):
        model = torch.load(f"{train_directory}/model.{epoch}.pth", weights_only=True)
        self.load_state_dict(model)
