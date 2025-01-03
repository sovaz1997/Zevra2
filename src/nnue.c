#include <stdio.h>
#include "nnue.h"

int isExists(Board* board, int color, int piece, int sq) {
    return !!(board->pieces[piece] & board->colours[color] & bitboardCell(sq));
}

int getInputIndexOf(int color, int piece, int sq) {
    return color * 64 * 6 + (piece - 1) * 64 + sq;
}

void setNNUEInput(NNUE* nnue, int index, int value) {
    int difference = value - nnue->inputs[index];
    nnue->inputs[index] += value;

    // TODO: recompute eval
}

void modifyNnue(NNUE* nnue, Board* board, int color, int piece) {
    for(int sq = 0; sq < 64; ++sq) {
        int weight = isExists(board, color, piece, sq);
        setNNUEInput(nnue, getInputIndexOf(color, piece, sq), weight);
    }
}

NNUE* createNNUE(Board* board) {
    NNUE* nnue = (NNUE*) malloc(sizeof(NNUE));

    for (int i = 0; i < INPUTS_COUNT; ++i) {
        nnue->inputs[i] = 0;
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

    int sum = 0;
    for (int i = 0; i < INPUTS_COUNT; ++i) {
        if (nnue->inputs[i] > 0) {
          // print index
          sum += i;
          printf("%d\n", i);
        }
    }
    printf("Sum: %d\n", sum);

    return nnue;
}