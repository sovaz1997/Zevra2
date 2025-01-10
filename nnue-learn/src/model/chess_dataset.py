import csv
import os
import struct

from torch.utils.data import IterableDataset, get_worker_info
import chess
from src.model.train_data_manager import TrainDataManager


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


def calculate_nnue_input_layer(board: chess.Board):
    nnue_input = 768 * [0]
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


def calculate_nnue_input_layer_cached(board_fen: str):
    return calculate_nnue_input_layer(chess.Board(board_fen))


def unpack_bits(packed_data, total_bits):
    unpacked = []
    for byte in packed_data:
        for i in range(7, -1, -1):
            unpacked.append((byte >> i) & 1)
    return unpacked[:total_bits]


def pack_bits(data):
    packed = bytearray()

    for i in range(0, len(data), 8):
        byte = 0
        for bit in data[i:i + 8]:
            byte = (byte << 1) | bit
        packed.append(byte)
    return packed


def save_nnue_data_bit_optimized(writer, nnue_input, eval_score):
    packed_input = pack_bits(nnue_input)
    writer.write(packed_input)
    writer.write(struct.pack('f', eval_score))


class ChessDataset(IterableDataset):
    def __init__(self, file_path, train_data_manager: TrainDataManager):
        print("Prepare dataset...")
        self.data_manager = train_data_manager
        self.input_size = train_data_manager.get_input_size()
        self.dataset_positions_count = 0
        self.packed_size = train_data_manager.get_packed_size()
        self.record_size = train_data_manager.get_record_size()
        self.bin_file_path = file_path + '.bin'
        print("Bin file path: ", self.bin_file_path)

        if os.path.exists(self.bin_file_path):
            print("Dataset already prepared")
            self.dataset_positions_count = os.path.getsize(self.bin_file_path) // self.record_size
            return

        with (open(file_path, 'r') as f):
            reader = csv.reader(f)

            with open(self.bin_file_path, 'wb') as writer:
                for idx, row in enumerate(reader):
                    if idx % 10000 == 0:
                        print(f"Processed positions: {idx}", flush=True)
                    fen, score = row
                    try:
                        save_nnue_data_bit_optimized(
                            writer,
                            calculate_nnue_input_layer_cached(fen),
                            float(score)
                        )
                        self.dataset_positions_count += 1
                    except Exception as e:
                        print(e)
                        continue
        print("Dataset prepared")

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            start = 0
            step = 1
        else:
            start = worker_info.id
            step = worker_info.num_workers

        with open(self.bin_file_path, 'rb') as f:
            for idx in range(start, self.dataset_positions_count, step):
                f.seek(idx * self.record_size)
                record = f.read(self.record_size)

                if not record:
                    break

                # packed_input = record[:self.packed_size]
                # eval_score = struct.unpack('f', record[self.packed_size:])[0]
                # nnue_input = unpack_bits(packed_input, self.input_size)
                yield self.data_manager.parse_record(record)
                # yield (
                #     tensor(nnue_input, dtype=float32),
                #     tensor(eval_score, dtype=float32),
                # )
