import chess
from torch import nn, clamp

from src.constants import SIMPLE_NETWORK_INPUT_SIZE
from src.model.nnue import NNUE
from src.utils import ctzll


def calculate_nnue_index(color: bool, piece: int, square: int, king_square: int):
    colors_mapper = {
        chess.WHITE: 0,
        chess.BLACK: 1
    }

    pieces_mapper = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
    }

    piece_index = pieces_mapper[piece] * 2 + colors_mapper[color]

    return square + (piece_index + king_square * 10) * 64


# def calculate_nnue_input_layer(board: chess.Board):
#     nnue_input_us = [0] * INPUT_SIZE
#     nnue_input_them = [0] * INPUT_SIZE
#
#     white_king_square = board.king(chess.WHITE)
#     black_king_square = board.king(chess.BLACK)
#
#     for color in chess.COLORS:
#         for piece in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
#             for square in range(64):
#                 if (board.piece_at(square) is not None
#                         and board.piece_at(square).piece_type == piece
#                         and board.piece_at(square).color == color):
#                     nnue_input_us[calculate_nnue_index(color, piece, square, white_king_square)] = 1
#                     nnue_input_them[calculate_nnue_index(not color, piece, square ^ 56, black_king_square ^ 56)] = 1
#
#     return nnue_input_us, nnue_input_them


class SimpleNetwork(NNUE):
    def __init__(self, hidden_size):
        super(SimpleNetwork, self).__init__()
        self.fc1 = nn.Linear(SIMPLE_NETWORK_INPUT_SIZE, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, 1, bias=False)

    def save_weights(self, epoch: int):
        super().save_weights(epoch)
        self._save_weight(self.fc1, "fc1", epoch)
        self._save_weight(self.fc2, "fc2", epoch)

    def forward(self, x):
        x = clamp(self.fc1(x), 0, 1)
        x = self.fc2(x)
        return x
