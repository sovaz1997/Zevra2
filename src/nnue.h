#ifndef NNUE_H
#define NNUE_H

#include "board.h"

static const int INPUTS_COUNT = 768;
static const int INNER_LAYER_COUNT = 256;
// int isNNUEEnabled = 0;


struct NNUE {
    double inputs[INPUTS_COUNT];
    double weights_1[INNER_LAYER_COUNT][INPUTS_COUNT];
    double weights_2[INNER_LAYER_COUNT];
    double accumulators[INNER_LAYER_COUNT];
    double eval;
};

NNUE* nnue;

double ReLU(double x);
int isExists(Board* board, int color, int piece, int sq);
int getInputIndexOf(int color, int piece, int sq);
void setNNUEInput(NNUE* nnue, int index);
void resetNNUE(NNUE* nnue);
void resetNNUEInput(NNUE* nnue, int index);
void modifyNnue(NNUE* nnue, Board* board, int color, int piece);
void initNNUEWeights();
void initNNUEPosition(NNUE* nnue, Board* board);
void loadNNUEWeights();
void debug_nnue_calculation(struct NNUE *nnue);

#endif