#ifndef NNUE_H
#define NNUE_H

#include "board.h"

#define PERSPECTIVE_MASK 56
#define INPUTS_COUNT 768
#define INNER_LAYER_COUNT 256
#define MAX_FEN_LENGTH 1000


struct NNUE {
    S16 inputs[INPUTS_COUNT];
    S16 inputs_perspective[INPUTS_COUNT];
    double weights_1[INNER_LAYER_COUNT][INPUTS_COUNT];
    S32 weights_1_quantized[INPUTS_COUNT][INNER_LAYER_COUNT];
    double weights_1_perspective[INNER_LAYER_COUNT][INPUTS_COUNT];
    S32 weights_1_perspective_quantized[INPUTS_COUNT][INNER_LAYER_COUNT];
    double weights_2[2 * INNER_LAYER_COUNT];
    S32 weights_2_quantized[2 * INNER_LAYER_COUNT];
    S32 accumulators[INNER_LAYER_COUNT];
    S32 accumulators_perspective[INNER_LAYER_COUNT];
    double eval;
};

NNUE* nnue;
char fen[MAX_FEN_LENGTH];

double ReLU(double x);
int isExists(Board* board, int color, int piece, int sq);
int getInputIndexOf(int color, int piece, int sq);
void resetNNUE(NNUE* nnue);
void setDirectNNUEInput(NNUE* nnue, int index);
void resetDirectNNUEInput(NNUE* nnue, int index);
void setPerspectiveNNUEInput(NNUE* nnue, int index);
void resetPerspectiveNNUEInput(NNUE* nnue, int index);
void modifyNnue(NNUE* nnue, Board* board, int color, int piece);
void recalculateEval(NNUE* nnue, int color);
void initNNUEPosition(NNUE* nnue, Board* board);
void loadNNUEWeights();
void debug_nnue_calculation(struct NNUE *nnue);
void resetNNUEInput(S16* inputs, S32* accumulators, S32 (*weights)[INNER_LAYER_COUNT], int index);
void setNNUEInput(S16* inputs, S32* accumulators, S32 (*weights)[INNER_LAYER_COUNT], int index);

#endif