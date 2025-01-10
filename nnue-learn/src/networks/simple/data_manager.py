import struct

import chess
from torch import tensor, float32

from src.networks.simple.constants import SIMPLE_NETWORK_INPUT_SIZE
from src.model.train_data_manager import TrainDataManager
from src.utils import ctzll, unpack_bits, pack_bits


def calculate_nnue_index(color: bool, piece: int, square: int):
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
        chess.KING: 5
    }

    return 64 * 6 * colors_mapper[color] + pieces_mapper[piece] * 64 + square

class SimpleNetworkDataManager(TrainDataManager):
    def calculate_nnue_input_layer(self, fen: str):
        board = chess.Board(fen)
        nnue_input = SIMPLE_NETWORK_INPUT_SIZE * [0]
        occupied = board.occupied
        piece_map = board.piece_map()

        while occupied:
            square = ctzll(occupied)
            piece = piece_map[square]
            color = piece.color
            piece_type = piece.piece_type
            nnue_input[calculate_nnue_index(color, piece_type, square)] = 1
            occupied &= occupied - 1

        return nnue_input

    def get_packed_size(self):
        return SIMPLE_NETWORK_INPUT_SIZE // 8

    def get_record_size(self):
        return self.get_packed_size() + 4

    def parse_record(self, record: bytes):
        packed_size = self.get_packed_size()
        packed_input = record[:self.get_packed_size()]
        eval_score = struct.unpack('f', record[packed_size:])[0]
        nnue_input = unpack_bits(packed_input, SIMPLE_NETWORK_INPUT_SIZE)

        return (
            tensor(nnue_input, dtype=float32),
            tensor(eval_score, dtype=float32),
        )

    def get_bin_folder(self):
        return "simple"

    def save_bin_data(self, writer, fen: str, eval_score: float):
        nnue_input = self.calculate_nnue_input_layer(fen)
        packed_input = pack_bits(nnue_input)
        writer.write(packed_input)
        writer.write(struct.pack('f', eval_score))
