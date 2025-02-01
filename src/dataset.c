#include "dataset.h"

int fensWrited = 0;

double DRAW_SCORE = 0.5;
double WHITE_WIN_SCORE = 1;
double BLACK_WIN_SCORE = 0;


void resetGame(Game* game) {
    game->positionsCount = 0;
}

void addPosition(Game* game, char* fen, int eval) {
    Position* position = &game->positions[game->positionsCount];
    strcpy(position->fen, fen);
    position->eval = eval;
    ++game->positionsCount;
}

void saveGameToFile(Game* game, FILE* file, double gameResult) {
    for (int i = 0; i < game->positionsCount; ++i) {
        Position* position = &game->positions[i];
        fprintf(file, "%s,%d,%.1f\n", position->fen, position->eval, gameResult);
    }
}

int getMovesCount(Board* board) {
    U16 moveList[256];
    movegen(board, moveList);
    U16* moveListPtr = moveList;
    Undo undo;

    int illegalCount = 0;
    while (*moveListPtr) {
        ++moveListPtr;
        // check that not check to us
        makeMove(board, *moveListPtr, &undo);
        if (inCheck(board, !board->color)) {
            ++illegalCount;
        }
        unmakeMove(board, *moveListPtr, &undo);
    }

    return moveListPtr - moveList - illegalCount;
}

void makeRandomMove(Board* board) {
  int movesCount = getMovesCount(board);

  if (movesCount == 0) {
    return;
  }

    U16 moveList[256];
    movegen(board, moveList);
    U16* moveListPtr = moveList;
    while (*moveListPtr) {
        ++moveListPtr;
    }

    int moveIndex = rand() % (moveListPtr - moveList);
    printf("Move index: %d\n", moveIndex);
    Undo undo;
    makeMove(board, moveList[moveIndex], &undo);

    if (inCheck(board, !board->color)) {
        unmakeMove(board, moveList[moveIndex], &undo);
        makeRandomMove(board);
    }
}

void runGame(Board* board, FILE* file) {
    setFen(board, startpos);
    resetGame(&game);

    printf("Moves count: %d\n", getMovesCount(board));
    for (int i = 0; i < 12; i++) {
        makeRandomMove(board);
    }

    TimeManager tm = createFixNodesTm(5000);
    U16 moveList[256];

    while(1) {
        movegen(board, moveList);
        int movesCount = getMovesCount(board);
        if (movesCount == 0) {
            printf("No moves\n");

            if (inCheck(board, WHITE)) {
                saveGameToFile(&game, file, BLACK_WIN_SCORE);
                return;
            }

            if (inCheck(board, BLACK)) {
                saveGameToFile(&game, file, WHITE_WIN_SCORE);
                return;
            }

            saveGameToFile(&game, file, DRAW_SCORE);
            return;
        }

        if (isDraw(board)) {
            saveGameToFile(&game, file, DRAW_SCORE);
            return;
        }

        SearchInfo info = iterativeDeeping(board, tm);
        Undo undo;

        // check that best move capture
        int isNotGoodPosition = MoveType(info.bestMove) == NORMAL_MOVE && board->squares[MoveTo(info.bestMove)]
            || MoveType(info.bestMove) == ENPASSANT_MOVE
            || MoveType(info.bestMove) == PROMOTION_MOVE
            || mateScore(info.eval);

        int turn = board->color == WHITE ? 1 : -1;
        makeMove(board, info.bestMove, &undo);

        if (inCheck(board, board->color)) {
            isNotGoodPosition = 1;
        }

        if (isNotGoodPosition) {
            continue;
        }

        char fen[256];
        getFen(board, fen);

        addPosition(&game, fen, info.eval * turn);
        printf("Positions writed: %d\n", ++fensWrited);
    }
}

void createDataset(Board* board, int gamesCount, int seed, char* fileName, char* logFile) {
    NNUE_ENABLED = 0;
    SHOULD_HIDE_SEARCH_INFO_LOGS = 1;
    FILE* file = fopen(fileName, "w");

    FILE* log = fopen(logFile, "w");

    srand(seed);

    for(int i = 0; i < gamesCount; ++i) {
        runGame(board, file);

            fprintf(log, "Games played: %d; positions writed: %d; Progress: %.2f%%\n",
                  i,
                  fensWrited,
                    (double)i / gamesCount * 100);
            fflush(log);
    }

    fclose(file);
    fclose(log);
    exit(0);
}