import chess
import chess.pgn


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

    print(f"Process completed. Total games: {game_count}", flush=True)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_path = "ccrl.pgn"
    output_file = "ccrl_positions.txt"
    process_large_pgn(file_path, output_file)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
