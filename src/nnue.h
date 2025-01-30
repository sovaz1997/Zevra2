#ifndef NNUE_H
#define NNUE_H

#include "board.h"

#define INPUTS_COUNT 768
#define INNER_LAYER_COUNT 128


struct NNUE {
    S16 inputs[INPUTS_COUNT];
    double weights_1[INNER_LAYER_COUNT][INPUTS_COUNT];
    S32 weights_1_quantized[INPUTS_COUNT][INNER_LAYER_COUNT];
    double weights_2[INNER_LAYER_COUNT];
    S32 weights_2_quantized[INNER_LAYER_COUNT];
    double accumulators[INNER_LAYER_COUNT];
    double biases_1[INNER_LAYER_COUNT];
    double biases_2;
    S32 biases_1_quantized[INNER_LAYER_COUNT];
    S32 biases_2_quantized;
    double eval;
};

NNUE* nnue;

double ReLU(double x);
int isExists(Board* board, int color, int piece, int sq);
int getInputIndexOf(int color, int piece, int sq);
void setNNUEInput(NNUE* nnue, int index);
void resetNNUE(NNUE* nnue);
void resetNNUEInput(NNUE* nnue, int index);
void recalculateEval(NNUE* nnue);
void loadNNUEWeights();
void debug_nnue_calculation(struct NNUE *nnue);
TimeManager createFixNodesTm(int nodes);
void dataset_gen(Board* board, int from, int to, char* filename);
void loadWeightsLayer(char* filename, double* weights, int rows, int cols);

#endif