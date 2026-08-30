#include "dataset.h"

// Definitions for globals declared extern in dataset.h
Game game;
char fen_for_save[256];
U16 moveList[256];

int fensWrited = 0;
int datagenNodes = 5000; // nodes/move for labels; override via ZEVRA_GEN_NODES

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
    // bullet text format: "<FEN> | <white-relative cp> | <white-relative result>"
    for (int i = 0; i < game->positionsCount; ++i) {
        Position* position = &game->positions[i];
        fprintf(file, "%s | %d | %.1f\n", position->fen, position->eval, gameResult);
    }
}

int getMovesCount(Board* board) {
    U16 moveList[256];
    movegen(board, moveList);
    Undo undo;

    int legal = 0;
    for (U16* p = moveList; *p; ++p) {
        makeMove(board, *p, &undo);
        if (!inCheck(board, !board->color))
            ++legal;
        unmakeMove(board, *p, &undo);
    }

    return legal;
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

    for (int i = 0; i < 12; i++) {
        makeRandomMove(board);
    }

    TimeManager tm = createFixNodesTm(datagenNodes);

    while(1) {
        movegen(board, moveList);
        int movesCount = getMovesCount(board);
        if (movesCount == 0) {
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

        int turn = board->color == WHITE ? 1 : -1;

        // Keep only quiet, non-tactical positions so the label reflects a
        // static evaluation: skip if the side to move is in check, the best
        // move is a capture/en-passant/promotion, or the score is a mate.
        U64 occupancy = board->colours[WHITE] | board->colours[BLACK];

        // Derive material from the mailbox (squares[]) -- that is what getFen
        // writes, so it is the source of truth for the saved FEN. Skip bare-king
        // positions, and skip any position where the mailbox and the bitboards
        // disagree (a board desync bug can leave a phantom piece in the
        // bitboards in long endgames -- such labels would not match the FEN).
        int mailboxTotal = 0, mailboxNonKing = 0;
        for (int sq = 0; sq < 64; ++sq) {
            if (board->squares[sq]) {
                ++mailboxTotal;
                if (pieceType(board->squares[sq]) != KING)
                    ++mailboxNonKing;
            }
        }
        int badPosition = mailboxNonKing == 0 || mailboxTotal != (int) popcount(occupancy);

        int noisy = inCheck(board, board->color)
            || badPosition
            || (MoveType(info.bestMove) == NORMAL_MOVE && board->squares[MoveTo(info.bestMove)])
            || MoveType(info.bestMove) == ENPASSANT_MOVE
            || MoveType(info.bestMove) == PROMOTION_MOVE
            || mateScore(info.eval);

        // Save the position that was actually evaluated (before the move) with
        // its white-relative score. FEN and eval now correspond to each other.
        if (!noisy) {
            getFen(board, fen_for_save);
            addPosition(&game, fen_for_save, info.eval * turn);
            ++fensWrited;
        }

        makeMove(board, info.bestMove, &undo);
    }
}

void createDataset(Board* board, int gamesCount, int seed, char* fileName, char* logFile) {
    temperature = 0;
    SHOULD_HIDE_SEARCH_INFO_LOGS = 1;
    const char* nodesEnv = getenv("ZEVRA_GEN_NODES");
    if (nodesEnv)
        datagenNodes = atoi(nodesEnv);
    reallocTT(16); // small TT is plenty at datagen node counts; keeps memory low
                   // when running many workers in parallel (16MB vs 256MB each)
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

TimeManager createFixNodesTm(int nodes) {
    TimeManager tm = initTM();
    tm.nodes = nodes;
    tm.depth = 100;
    tm.searchType = FixedNodes;
    return tm;
}