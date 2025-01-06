import os

from torch.utils.data import Dataset, DataLoader, IterableDataset

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
        # self.data = pd.read_csv(file_path)
        # self.board = chess.Board()
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                fen, score = row
                board = chess.Board(fen)
                inputs = calculate_nnue_input_layer(board)
                yield torch.tensor(inputs, dtype=torch.float32), torch.tensor(float(score), dtype=torch.float32)

    # def __len__(self):
    #     return len(self.data)

    # def __getitem__(self, idx):
    #     fen = self.data.iloc[idx, 0]
    #     score = self.data.iloc[idx, 1] # / 1000
    #     self.board.set_fen(fen)
    #     inputs = calculate_nnue_input_layer(self.board)
    #     return torch.tensor(inputs, dtype=torch.float32), torch.tensor(score, dtype=torch.float32)


class NNUE(nn.Module):
    def __init__(self, input_size=768, hidden_size=256, output_size=1):
        super(NNUE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.fc2(x)
        return x


# def save_layer_weights(weights: nn.Linear, filename):
#     with open(filename, 'w') as file:
#         weights = weights.weight.data.numpy()
#         for row in weights:
#             file.write(','.join([str(x) for x in row]) + '\n')

def save_layer_weights(weights: nn.Linear, filename):
    weight_matrix = weights.weight.cpu().data.numpy()  # shape [out_features, in_features]

    flat_weights = weight_matrix.flatten()  # shape [out_features * in_features]

    with open(filename, 'w') as file:
        file.write(','.join(str(x) for x in flat_weights))
        file.write('\n')


def save_nnue_weights(net: NNUE, epoch: int):
    save_layer_weights(net.fc1, f"fc1.{epoch}.weights.csv")
    save_layer_weights(net.fc2, f"fc2.{epoch}.weights.csv")
    # save_layer_weights(net.fc3, "fc3.weights.csv")


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


def evaluate_test_fen(model, test_fen: str):
    board = chess.Board(test_fen)
    # print board
    print(board)

    # Превращаем доску в входные данные
    nnue_input = calculate_nnue_input_layer(board)  # Это ваш метод из кода
    nnue_input_tensor = torch.tensor(nnue_input, dtype=torch.float32).unsqueeze(0)

    # Переключаем модель в режим оценки
    model.eval()
    with torch.no_grad():
        output = model(nnue_input_tensor)
        score = output.item() * 1000

    # debug_nnue_calculation(model, nnue_input_tensor.squeeze(0))

    return score


def debug_nnue_calculation(model: nn.Module, input_vector: torch.Tensor):
    """
    Выводит пошаговый расчёт сети с архитектурой:
      fc1(768->256, bias=False) -> ReLU -> fc2(256->1, bias=False).
    model        : уже инициализированная модель NNUE (weights загружены)
    input_vector : 1D-тензор формы [768], содержащий вход (чаще всего 0/1)
    """

    # Если input_vector имеет размер [1, 768], можно привести к [768]:
    if input_vector.dim() == 2 and input_vector.shape[0] == 1:
        input_vector = input_vector.squeeze(0)  # [768]

    # Инпуты
    # print("\n=== Входной вектор ===")
    # for i in range(768):
    #     if input_vector[i] != 0.0:
    #         print(f"input[{i}] = {input_vector[i].item()}")

    with torch.no_grad():
        # ---------------------------
        # 1) Скрытый слой: fc1 -> ReLU
        # ---------------------------
        # Форма матрицы weight у fc1: [256, 768]
        # т.е. fc1.weight[j, i] это вес к j-му нейрону от i-го входа
        fc1_weights = model.fc1.weight  # shape = [256, 768]

        # Подготовим массив (или список) для хранения выхода скрытого слоя до ReLU
        hidden_raw = [0.0] * 256
        hidden = [0.0] * 256  # после ReLU

        print("=== Скрытый слой (fc1 -> ReLU) ===")
        for j in range(256):
            sum_val = 0.0
            partial_contribs = []

            # Считаем вклад только тех i, где input[i] != 0
            for i in range(768):
                x_i = input_vector[i].item()
                if x_i != 0.0:
                    w_ji = fc1_weights[j, i].item()

                    contrib = x_i * w_ji
                    if contrib != 0.0:
                        sum_val += contrib
                        partial_contribs.append((i, x_i, w_ji, contrib))

            hidden_raw[j] = sum_val
            # Применяем ReLU
            relu_val = max(0.0, sum_val)
            hidden[j] = relu_val

            # Если после ReLU что-то осталось > 0, распечатаем подробнее
            if relu_val > 0.0:
                print(f"\n[Нейрон {j}] Сумма до ReLU = {sum_val:.4f}, после ReLU = {relu_val:.4f}")
                print("  Вклады (i, input[i], weight, contrib):")
                for (i_idx, x_val, w_val, c_val) in partial_contribs:
                    print(f"    i={i_idx}, x={x_val:.1f}, w={w_val:.4f}, contrib={c_val:.4f}")

        # ---------------------------
        # 2) Выходной слой: fc2
        # ---------------------------
        # Форма weight у fc2: [1, 256]
        fc2_weights = model.fc2.weight[0]  # shape = [256], т.к. output_size=1

        sum_out = 0.0
        partial_out = []

        print("\n=== Выходной слой (fc2) ===")
        # Суммируем вклад от каждого нейрона скрытого слоя
        for j in range(256):
            h_j = hidden[j]  # выход j-го нейрона после ReLU
            if h_j != 0.0:  # если нейрон не затух
                w_j = fc2_weights[j].item()
                c = h_j * w_j
                if c != 0.0:
                    sum_out += c
                    partial_out.append((j, h_j, w_j, c))

        print(f"Сумма на выходе (до умножения на 1000): {sum_out:.4f}")

        if len(partial_out) > 0:
            print("  Вклады активных нейронов скрытого слоя (j, hidden[j], weight, contrib):")
            for (j_idx, h_val, w_val, c_val) in partial_out:
                print(f"    j={j_idx}, hidden={h_val:.4f}, w={w_val:.4f}, contrib={c_val:.4f}")

        # Предположим, что при обучении лейблы делились на 1000
        # Тогда умножаем выход, чтобы получить "сантипешки".
        scaled_out = sum_out * 1000.0

        print(f"\nИтоговый выход (сырое) = {sum_out:.4f}")
        print(f"Итоговая оценка (умножено на 1000) = {scaled_out:.1f} cp\n")


if __name__ == '__main__':
    model = NNUE()
    device = torch.device("mps")
    model = model.to(device)

    dataset = ChessDataset("100millions_dataset.csv")
    dataloader = DataLoader(dataset, batch_size=512, num_workers=12, pin_memory=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    epoch = load_checkpoint(model, optimizer, scheduler)

    print(model)
    # print(evaluate_test_fen(model, "1qqqk3/1qqqp3/1qqq4/1qqq4/8/R7/3Q4/3QK3 w HAha - 0 1"))
    # print(evaluate_test_fen(model, "rnbqkbnr/ppp3pp/8/4p3/3pNp2/3P1N2/PPP1PPPP/R1BQKB1R b KQkq - 1 6"))
    # print(evaluate_test_fen(model, "rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQq - 0 1"))

    while True:
        model.train()
        running_loss = 0.0
        index = 0
        for (batch_inputs, batch_scores) in dataloader:
            index += 1
            batch_inputs = batch_inputs.to(device)
            batch_scores = batch_scores.to(device)

            if index % 10 == 0:
                print(f"Learning: {index}")
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs.squeeze(), batch_scores)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # if index % 10 == 0:
                # print(f"Learning: {index}")
        loss = running_loss / len(dataloader)
        scheduler.step(loss)
        save_nnue_weights(model, epoch)

        print(f"Epoch [{epoch}], Loss: {running_loss / len(dataloader):.4f}", flush=True)
        save_checkpoint(model, optimizer, scheduler, epoch)
        epoch += 1

        if loss < 0.05:
            break
        # print LR
        print(optimizer.param_groups[0]['lr'])
