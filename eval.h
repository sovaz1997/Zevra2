#ifndef EVAL_H
#define EVAL_H

#include "board.h"
#include "psqt.h"

//Веса фигур
enum figureWeights {
    PAWN_EV = 100,
    KNIGHT_EV = 400,
    BISHOP_EV = 430,
    ROOK_EV = 650,
    QUEEN_EV = 1100
};

int fullEval(Board* board);
int materialEval(Board* board);
int psqtEval(Board* board);
int psqtPieceEval(Board* board, U64 mask, const int* pstTable);

#endif