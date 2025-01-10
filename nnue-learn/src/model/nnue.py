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
    def _save_weight(self, layer: nn.Linear, name: str, epoch: int):
        save_layer_weights(layer, f"{name}.{epoch}.weights.csv")

    def save_weights(self, epoch: int):
        model = self.state_dict()
        torch.save(model, f"model.{epoch}.pth")

    def load_weights(self, epoch: int):
        model = torch.load(f"model.{epoch}.pth", weights_only=True)
        self.load_state_dict(model)
