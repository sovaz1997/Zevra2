import os

from torch.utils.data import Dataset, DataLoader

import chess
import chess.pgn
import chess.engine
import csv
import torch
import torch.nn as nn
import pandas as pd


DATASET_POSITIONS_COUNT = 250000


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
    print()


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


class ChessDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.board = chess.Board()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen = self.data.iloc[idx, 0]
        score = self.data.iloc[idx, 1] / 1000
        self.board.set_fen(fen)
        inputs = calculate_nnue_input_layer(self.board)
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(score, dtype=torch.float32)



class NNUE(nn.Module):
    def __init__(self, input_size=768, hidden_size=8, output_size=1):
        super(NNUE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def save_layer_weights(weights: nn.Linear, filename):
    with open(filename, 'w') as file:
        # weights = weights.weight.cpu().data.numpy()
        weights = weights.weight.data.numpy()
        for row in weights:
            file.write(','.join([str(x) for x in row]) + '\n')


def save_nnue_weights(net: NNUE):
    save_layer_weights(net.fc1, "fc1.weights.csv")
    save_layer_weights(net.fc2, "fc2.weights.csv")
    save_layer_weights(net.fc3, "fc3.weights.csv")


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
        return 1
    checkpoint = torch.load(filename, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']


if __name__ == '__main__':
    model = NNUE()

    dataset = ChessDataset("dataset.csv")
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    epoch = load_checkpoint(model, optimizer, scheduler)

    while True:
        model.train()
        running_loss = 0.0
        index = 0
        for (batch_inputs, batch_scores) in dataloader:
            index += 1
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs.squeeze(), batch_scores)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if index % 10 == 0:
                print(f"Learning: {index}")
        loss = running_loss / len(dataloader)
        scheduler.step(loss)
        save_nnue_weights(model)
        print(f"Epoch [{epoch}], Loss: {running_loss / len(dataloader):.4f}", flush=True)
        save_checkpoint(model, optimizer, scheduler, epoch)
        epoch += 1


        if loss < 0.2:
            break
        # print LR
        print(optimizer.param_groups[0]['lr'])