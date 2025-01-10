# import os
#
# from torch.utils.data import DataLoader, IterableDataset
#
# import chess
# import chess.pgn
# import chess.engine
# import csv
# import torch
# import torch.nn as nn
#
# DATASET_POSITIONS_COUNT = 1000000000000
# HIDDEN_SIZE = 128
# INPUT_SIZE = 40960
#
#
# def process_large_pgn(file_path, output_file):
#     with open(file_path, 'r') as pgn_file, open(output_file, 'w') as fen_file:
#         game_count = 0
#         while True:
#             game = chess.pgn.read_game(pgn_file)
#             if game is None:
#                 break
#
#             board = game.board()
#             for move in game.mainline_moves():
#                 board.push(move)
#                 fen_file.write(board.fen() + '\n')
#
#             game_count += 1
#             if game_count % 1000 == 0:
#                 print(f"Processed games: {game_count}", flush=True)
#
#     print(f"Process completed. Total games: {game_count}")
#
#
# def print_bitboard(bitboard):
#     for rank in range(8, 0, -1):
#         line = ""
#         for file in range(1, 9):
#             square = chess.square(file - 1, rank - 1)
#             if bitboard & chess.BB_SQUARES[square]:
#                 line += "1 "
#             else:
#                 line += ". "
#         print(line)
#
#
# def calculate_nnue_index(color: bool, piece: int, square: int, king_square: int):
#     colors_mapper = {
#         chess.WHITE: 0,
#         chess.BLACK: 1
#     }
#
#     pieces_mapper = {
#         chess.PAWN: 0,
#         chess.KNIGHT: 1,
#         chess.BISHOP: 2,
#         chess.ROOK: 3,
#         chess.QUEEN: 4,
#     }
#
#     piece_index = pieces_mapper[piece] * 2 + colors_mapper[color]
#
#     return square + (piece_index + king_square * 10) * 64
#
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
#
#
#
# def analyse_position(board: chess.Board, engine: chess.engine.SimpleEngine):
#     try:
#         result = engine.analyse(board, chess.engine.Limit(nodes=10000))
#         score = result["score"].white().score(mate_score=100000)
#         return score
#     except Exception as e:
#         engine.quit()
#         return None
#
#
# def read_fens(file_path: str):
#     with open(file_path, 'r') as file:
#         for line in file:
#             yield line.strip()
#
#
# def evaluate_positions(file_path: str, output_csv_path: str):
#     board = chess.Board()
#
#     engine = chess.engine.SimpleEngine.popen_uci("./zevra")
#
#     positions_count = 0
#     with open(output_csv_path, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["FEN", "Score"])
#
#         for fen in read_fens(file_path):
#             try:
#                 board.set_fen(fen)
#                 eval = analyse_position(board, engine)
#                 positions_count += 1
#                 if positions_count % 5000 == 0:
#                     print(f"Processed positions: {positions_count}", flush=True)
#                 # print(f"FEN: {fen}, Evaluation: {eval}")
#                 if eval is not None:
#                     writer.writerow([fen, eval])
#                 if positions_count > DATASET_POSITIONS_COUNT:
#                     engine.close()
#                     return
#             except Exception as e:
#                 # print(e)
#                 engine.close()
#                 engine = chess.engine.SimpleEngine.popen_uci("./zevra")
#
#
# class ChessDataset(IterableDataset):
#     def __init__(self, file_path):
#         # self.data = pd.read_csv(file_path)
#         # self.board = chess.Board()
#         self.file_path = file_path
#
#     def __iter__(self):
#         worker_info = torch.utils.data.get_worker_info()
#
#         if worker_info is None:
#             start = 0
#             step = 1
#         else:
#             # Split workload among workers
#             start = worker_info.id
#             step = worker_info.num_workers
#
#         with (open(self.file_path, 'r') as f):
#             reader = csv.reader(f)
#             for idx, row in enumerate(reader):
#                 if idx >= DATASET_POSITIONS_COUNT:
#                      break
#
#                 if idx % step == start:
#                     fen, score = row
#                     try:
#                         board = chess.Board(fen)
#                         # turn = torch.tensor(1 if board.turn == chess.WHITE else 0, dtype=torch.float32)
#                         # side_multiplier = 1 if board.turn == chess.WHITE else -1
#                         side_multiplier = 1
#                         input1, input2 = calculate_nnue_input_layer(board)
#                         yield (
#                             torch.tensor(input1, dtype=torch.float32),
#                             torch.tensor(input2, dtype=torch.float32),
#                             torch.tensor(float(score) * side_multiplier, dtype=torch.float32),
#                         )
#                     except Exception as e:
#                         print(e)
#                         continue
#
#
# class NNUE(nn.Module):
#     def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=1):
#         super(NNUE, self).__init__()
#         self.fc1_us = nn.Linear(input_size, hidden_size, bias=False)
#         self.fc1_them = nn.Linear(input_size, hidden_size, bias=False)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(2 * hidden_size, output_size, bias=False)
#
#     def forward(self, x1, x2):
#         def print_acc(acc):
#             quant = torch.round(acc * 255)
#
#             for(i, val) in enumerate(quant.data.numpy()[0]):
#                 if val > 0:
#                     print(f"{i}: {val}")
#
#         first_input = self.fc1_us(x1)
#         second_input = self.fc1_them(x2)
#
#         acc1 = torch.clamp(first_input, 0, 1)
#         acc2 = torch.clamp(second_input, 0, 1)
#
#         # x255 and round
#         # quant = torch.round(acc1 * 255)
#         # quant.data.numpy()
#         # for(i, val) in enumerate(quant[0]):
#         #     print(f"{i}: {val}")
#
#         # side = turn.unsqueeze(1).float()
#         # x = (side * torch.cat((acc1, acc2), 1)) + ((1 - side) * torch.cat((acc2, acc1), 1))
#         x = torch.cat((acc1, acc2), 1)
#
#         return self.fc2(x)
#
# def save_layer_weights(weights: nn.Linear, filename):
#     weight_matrix = weights.weight.cpu().data.numpy()  # shape [out_features, in_features]
#
#     flat_weights = weight_matrix.flatten()  # shape [out_features * in_features]
#
#     with open(filename, 'w') as file:
#         file.write(','.join(str(x) for x in flat_weights))
#         file.write('\n')
#
#
# def save_nnue_weights(net: NNUE, epoch: int):
#     save_layer_weights(net.fc1_us, f"fc1_us.{epoch}.weights.csv")
#     save_layer_weights(net.fc1_them, f"fc1_them.{epoch}.weights.csv")
#     save_layer_weights(net.fc2, f"fc2.{epoch}.weights.csv")
#
#
# def save_checkpoint(
#         model,
#         optimizer,
#         scheduler,
#         epoch,
#         filename="checkpoint.pth"):
#     checkpoint = {
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'scheduler_state_dict': scheduler.state_dict(),
#         'epoch': epoch
#     }
#     torch.save(checkpoint, filename)
#     print(f"Checkpoint saved at epoch {epoch}")
#
#
# def load_checkpoint(
#         model,
#         optimizer,
#         scheduler,
#         filename="checkpoint.pth"):
#     if not os.path.exists(filename):
#         return 1
#     checkpoint = torch.load(filename, weights_only=True)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#     return checkpoint['epoch']
#
#
# def evaluate_test_fen(model, test_fen: str):
#     board = chess.Board(test_fen)
#     # print board
#     print(board)
#
#     # Превращаем доску в входные данные
#     nnue_input_us, nnue_input_them = calculate_nnue_input_layer(board)  # Это ваш метод из кода
#     nnue_input_tensor_us = torch.tensor(nnue_input_us, dtype=torch.float32).unsqueeze(0)
#     nnue_input_tensor_them = torch.tensor(nnue_input_them, dtype=torch.float32).unsqueeze(0)
#
#     # Переключаем модель в режим оценки
#     model.cpu().eval()
#     with torch.no_grad():
#         output = model(nnue_input_tensor_us, nnue_input_tensor_them)
#         score = output.item()
#
#     return score
#
#
# def train():
#     model = NNUE()
#     device = torch.device("mps")
#     model = model.to(device)
#
#     dataset = ChessDataset("100millions_dataset.csv")
#     dataloader = DataLoader(dataset, batch_size=512, num_workers=11, prefetch_factor=2)
#
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
#     epoch = load_checkpoint(model, optimizer, scheduler)
#
#     print(model)
#     # print(evaluate_test_fen(model, "qqqqkqqr/ppqqqppp/8/8/8/8/6P1/4K3 w HAka - 0 1"))
#     # print(evaluate_test_fen(model, "4k3/pppppppp/8/8/8/8/4P3/4K3 w - - 0 1"))
#     # print(evaluate_test_fen(model, "4k3/pppppppp/8/8/8/8/4P3/4K3 b - - 0 1"))
#     # print(evaluate_test_fen(model, "rnbqkbnr/ppp3pp/8/4p3/3pNp2/3P1N2/PPP1PPPP/R1BQKB1R b KQkq - 1 6"))
#     # print(evaluate_test_fen(model, "1nbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1"))
#     # print(evaluate_test_fen(model, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQq - 0 1"))
#     # print(evaluate_test_fen(model, "1nb1kbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1"))
#     # print(evaluate_test_fen(model, "1nb1kbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQ - 0 1"))
#     # print(evaluate_test_fen(model, "4k3/7p/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1"))
#     # print(evaluate_test_fen(model, "r1b1k2r/pppp1ppp/8/4p3/2P1P3/P2P1qn1/R7/2B3K1 w kq - 0 19"))
#     # print(evaluate_test_fen(model, "r1bqkb1r/pppp1ppp/2n5/4p2n/4P3/2NP4/PPP2PPP/R1B1KBNR w KQkq - 0 5"))
#     # print(evaluate_test_fen(model, "3qk3/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1"))
#     # print(evaluate_test_fen(model, "2k5/ppp3pp/2nrp3/8/2Rb3B/8/2P2PPP/2K5 b - - 6 28"))
#
#     while True:
#         model.train()
#         running_loss = 0.0
#         count = 1
#         index = 0
#         # for (batch_inputs, batch_scores) in dataloader:
#         for batch_idx, (batch_inputs_us, batch_inputs_them, batch_scores) in enumerate(dataloader):
#             index += 1
#             if index % 5000 == 0:
#                 print(f"Learning: {index}")
#             count += len(batch_inputs_us)
#             batch_inputs_us = batch_inputs_us.to(device, non_blocking=True)
#             batch_inputs_them = batch_inputs_them.to(device, non_blocking=True)
#             batch_scores = batch_scores.to(device, non_blocking=True)
#
#             optimizer.zero_grad()
#             outputs = model(batch_inputs_us, batch_inputs_them)
#             loss = criterion(outputs.squeeze(), batch_scores)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#         loss = (running_loss / count)
#         scheduler.step(loss)
#         save_nnue_weights(model, epoch)
#
#         print(f"Epoch [{epoch}], Loss: {loss:.4f}", flush=True)
#         save_checkpoint(model, optimizer, scheduler, epoch)
#         epoch += 1
#
#         if loss < 0.05:
#             break
#         print(optimizer.param_groups[0]['lr'])
#
# if __name__ == '__main__':
#     train()
import os

from torch.utils.data import DataLoader
from src.model.chess_dataset import ChessDataset
from src.model.managers.simple_network_data_manager import SimpleNetworkDataManager
from src.model.networks.simple_network import SimpleNetwork
from src.model.train_data_manager import TrainDataManager
from src.train import train

main_dir = os.path.dirname(os.path.abspath(__file__))

TRAIN_DATASET_PATH = os.path.join(main_dir, "train_100millions_dataset.csv")
VALIDATION_DATASET_PATH = os.path.join(main_dir, "validate_100millions_dataset.csv")

def create_data_loader(manager: TrainDataManager, path: str):
    dataset = ChessDataset(path, manager)
    return DataLoader(dataset, batch_size=512, num_workers=11, persistent_workers=True, prefetch_factor=2)


def run_train_nnue(
        hidden_size: int,
        train_dataset_path: str,
        validation_dataset_path: str,
):
    manager = SimpleNetworkDataManager()

    train(
        SimpleNetwork(hidden_size),
        create_data_loader(manager, train_dataset_path),
        create_data_loader(manager, validation_dataset_path)
    )

if __name__ == '__main__':
    run_train_nnue(128, TRAIN_DATASET_PATH, VALIDATION_DATASET_PATH)