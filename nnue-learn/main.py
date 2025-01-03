import chess
import chess.pgn
import chess.engine


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

    sum = 0

    for i in range(len(nnue_input)):
        if nnue_input[i] == 1:
            sum += i
            print(f"Index: {i}, Value: 1")

    print(f"Sum: {sum}")


def evaluate_positions():
    with chess.engine.SimpleEngine.popen_uci("./zevra") as engine:
        board = chess.Board()

        # Печатаем текущую позицию
        print("Current position:")
        print(board)

        # Запрашиваем лучший ход
        result = engine.analyse(board, chess.engine.Limit(nodes=10000))
        evaluation = result["score"].white()
        print(evaluation)
        # print bitboard
        print_bitboard(board.bishops & board.occupied_co[chess.WHITE])

        calculate_nnue_input_layer(board)

        for square in range(0, 64):
            if (chess.BB_SQUARES[square] & board.bishops & board.occupied_co[chess.WHITE]):
                print(f"Bishop on square {square} is on white's turn.")


if __name__ == '__main__':
    print('Starting')
    evaluate_positions()
    # export pgn to fen
    # file_path = "ccrl.pgn"
    # output_file = "ccrl_positions.txt"
    # process_large_pgn(file_path, output_file)
