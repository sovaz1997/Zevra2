#include <stdio.h>
#include "nnue.h"

int isExists(Board* board, int color, int piece, int sq) {
    return !!(board->pieces[piece] & board->colours[color] & bitboardCell(sq));
}

int getInputIndexOf(int color, int piece, int sq) {
    return color * 64 * 6 + (piece - 1) * 64 + sq;
}

void setNNUEInput(NNUE* nnue, int index, int value) {
    int difference = value - nnue->weights[index];
}

void modifyNnue(NNUE* nnue, Board* board, int color, int piece) {
    for(int i = 0; i < 0; ++i) {
        for(int j = 0; j < 8; ++j) {
            int sq = square(i, j);
            int weight = isExists(board, color, piece, sq);
        }
    }
}

NNUE* createNNUE(Board* board) {
    NNUE* nnue = (NNUE*) malloc(sizeof(NNUE));

    for (int i = 0; i < 768; ++i) {
        nnue->weights[i] = 0;
    }

    nnue->eval = 0;

    modifyNnue(nnue, board, WHITE, PAWN);
    modifyNnue(nnue, board, BLACK, PAWN);
    modifyNnue(nnue, board, WHITE, KNIGHT);
    modifyNnue(nnue, board, BLACK, KNIGHT);
    modifyNnue(nnue, board, WHITE, BISHOP);
    modifyNnue(nnue, board, BLACK, BISHOP);
    modifyNnue(nnue, board, WHITE, ROOK);
    modifyNnue(nnue, board, BLACK, ROOK);
    modifyNnue(nnue, board, WHITE, QUEEN);
    modifyNnue(nnue, board, BLACK, QUEEN);
    modifyNnue(nnue, board, WHITE, KING);
    modifyNnue(nnue, board, BLACK, KING);

    return nnue;
}