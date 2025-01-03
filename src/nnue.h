#include "board.h"

static const int INPUTS_COUNT = 768;
static const int INNER_LAYER_COUNT = 8;
//const int ACCUMULATOR_COUNT = 1024;

//const double WEIGHTS_1[INPUTS_COUNT][INNER_LAYER_COUNT];
//const double WEIGHTS_2[INNER_LAYER_COUNT][INNER_LAYER_COUNT];
//const double WEIGHTS_3[INNER_LAYER_COUNT];

struct NNUE {
    double inputs[INPUTS_COUNT];
    double weights_1[INPUTS_COUNT][INNER_LAYER_COUNT];
    double weights_2[INNER_LAYER_COUNT][INNER_LAYER_COUNT];
    double weights_3[INNER_LAYER_COUNT];
    double accumulator1[INNER_LAYER_COUNT];
    double accumulator2[INNER_LAYER_COUNT];
    double eval;
};

int isExists(Board* board, int color, int piece, int sq);
int getInputIndexOf(int color, int piece, int sq);
NNUE* createNNUE(Board* board);
