import os

from torch.utils.data import DataLoader, IterableDataset

import chess
import chess.pgn
import chess.engine
import csv
import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, filename="log.log",filemode="w")


VALIDATION_DATASET_PATH = "validate_100millions_dataset.csv"
TRAIN_DATASET_PATH = "train_100millions_dataset.csv"
DATASET_POSITIONS_COUNT = 1000000000000
HIDDEN_SIZE = 128
INPUT_SIZE = 768


def process_large_pgn(file_path, output_file):
    with open(file_path, 'r') as pgn_file, open(output_file, 'w') as fen_file:
        game_count = 0
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                fen_file.write(board.fen() + '\n')

            game_count += 1
            if game_count % 1000 == 0:
                print(f"Processed games: {game_count}", flush=True)

    print(f"Process completed. Total games: {game_count}")


def print_bitboard(bitboard):
    for rank in range(8, 0, -1):
        line = ""
        for file in range(1, 9):
            square = chess.square(file - 1, rank - 1)
            if bitboard & chess.BB_SQUARES[square]:
                line += "1 "
            else:
                line += ". "
        print(line)


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
    nnue_input = [0] * 768

    for color in chess.COLORS:
        for piece in chess.PIECE_TYPES:
            for square in range(64):
                if (board.piece_at(square) is not None
                        and board.piece_at(square).piece_type == piece
                        and board.piece_at(square).color == color):
                    nnue_input[calculate_nnue_index(color, piece, square)] = 1

    return nnue_input



def analyse_position(board: chess.Board, engine: chess.engine.SimpleEngine):
    try:
        result = engine.analyse(board, chess.engine.Limit(nodes=10000))
        score = result["score"].white().score(mate_score=100000)
        return score
    except Exception as e:
        engine.quit()
        return None


def read_fens(file_path: str):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()


def evaluate_positions(file_path: str, output_csv_path: str):
    board = chess.Board()

    engine = chess.engine.SimpleEngine.popen_uci("./zevra")

    positions_count = 0
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["FEN", "Score"])

        for fen in read_fens(file_path):
            try:
                board.set_fen(fen)
                eval = analyse_position(board, engine)
                positions_count += 1
                if positions_count % 100 == 0:
                    print(f"Processed positions: {positions_count}", flush=True)
                # print(f"FEN: {fen}, Evaluation: {eval}")
                if eval is not None:
                    writer.writerow([fen, eval])
                if positions_count > DATASET_POSITIONS_COUNT:
                    engine.close()
                    return
            except Exception as e:
                # print(e)
                engine.close()
                engine = chess.engine.SimpleEngine.popen_uci("./zevra")


class ChessDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            start = 0
            step = 1
        else:
            # Split workload among workers
            start = worker_info.id
            step = worker_info.num_workers

        with (open(self.file_path, 'r') as f):
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx >= DATASET_POSITIONS_COUNT:
                     break

                if idx % step == start:
                    fen, score = row
                    try:
                        board = chess.Board(fen)
                        # turn = torch.tensor(1 if board.turn == chess.WHITE else 0, dtype=torch.float32)
                        # side_multiplier = 1 if board.turn == chess.WHITE else -1
                        side_multiplier = 1
                        input1 = calculate_nnue_input_layer(board)
                        yield (
                            torch.tensor(input1, dtype=torch.float32),
                            torch.tensor(float(score) * side_multiplier, dtype=torch.float32),
                        )
                    except Exception as e:
                        print(e)
                        continue




class NNUE(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, output_size=1):
        super(NNUE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        x = torch.clamp(self.fc1(x), 0, 1)
        x = self.fc2(x)
        return x

def save_layer_weights(weights: nn.Linear, filename):
    weight_matrix = weights.weight.cpu().data.numpy()  # shape [out_features, in_features]

    flat_weights = weight_matrix.flatten()  # shape [out_features * in_features]

    with open(filename, 'w') as file:
        file.write(','.join(str(x) for x in flat_weights))
        file.write('\n')


def save_nnue_weights(net: NNUE, epoch: int):
    save_layer_weights(net.fc1, f"fc1.{epoch}.weights.csv")
    save_layer_weights(net.fc2, f"fc2.{epoch}.weights.csv")


def save_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch,
        filename="checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(
        model,
        optimizer,
        scheduler,
        filename="checkpoint.pth"):
    if not os.path.exists(filename):
        return 0
    checkpoint = torch.load(filename, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']

def validate_net(net: NNUE):
    dataset = ChessDataset(VALIDATION_DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=1, prefetch_factor=2)
    batches_length = 0
    criterion = nn.MSELoss()
    running_loss = 0.0

    for batch_idx, (batch_inputs, batch_scores) in enumerate(dataloader):
        batches_length += 1
        batch_inputs = batch_inputs.to("mps")
        batch_scores = batch_scores.to("mps")
        outputs = net(batch_inputs)
        loss = criterion(outputs.squeeze(), batch_scores)
        running_loss += loss.item()

    return running_loss / batches_length



def evaluate_test_fen(model, test_fen: str):
    board = chess.Board(test_fen)
    # print board
    print(board)

    # Превращаем доску в входные данные
    nnue_input = calculate_nnue_input_layer(board)  # Это ваш метод из кода
    nnue_input_tensor = torch.tensor(nnue_input, dtype=torch.float32).unsqueeze(0)

    # Переключаем модель в режим оценки
    model.cpu().eval()
    with torch.no_grad():
        output = model(nnue_input_tensor)
        score = output.item()

    return score


def train():
    model = NNUE()
    device = torch.device("mps")
    model = model.to(device)

    dataset = ChessDataset(TRAIN_DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=512, num_workers=1, prefetch_factor=2)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    epoch = load_checkpoint(model, optimizer, scheduler) + 1

    # model.to("cpu")
    # print(evaluate_test_fen(model, "6k1/5p2/3p2pp/2r5/1p5P/bQp3PB/P1R2PK1/3q4 b - - 1 40"))

    print(model)
    # validate_loss = validate_net(model)
    # print(f"Initial validate loss: {validate_loss:.4f}", flush=True)

    while True:
        model.train()
        running_loss = 0.0
        count = 0
        index = 0
        # for (batch_inputs, batch_scores) in dataloader:
        for batch_idx, (batch_inputs, batch_scores) in enumerate(dataloader):
            index += 1
            if index % 100 == 0:
                print(f"Learning: {index}")
            count += len(batch_inputs)
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_scores = batch_scores.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs.squeeze(), batch_scores)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        loss = (running_loss / index)
        scheduler.step(loss)
        save_nnue_weights(model, epoch)

        validate_loss = validate_net(model)
        print(f"Epoch [{epoch}], Train loss: {loss:.4f}, Validate loss: {validate_loss:.4f}", flush=True)
        logging.info(f"Epoch [{epoch}], Train loss: {loss:.4f}, Validate loss: {validate_loss:.4f}")
        save_checkpoint(model, optimizer, scheduler, epoch)
        epoch += 1

        if loss < 0.05:
            break
        print(optimizer.param_groups[0]['lr'])

if __name__ == '__main__':
    train()