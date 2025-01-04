#include <stdio.h>
#include "nnue.h"

int isExists(Board* board, int color, int piece, int sq) {
    return !!(board->pieces[piece] & board->colours[color] & bitboardCell(sq));
}

int getInputIndexOf(int color, int piece, int sq) {
    return color * 64 * 6 + (piece - 1) * 64 + sq;
}

double ReLU(double x) {
    return x > 0 ? x : 0;
}

int counter = 0;

void setNNUEInput(NNUE* nnue, int index) {
    if (nnue->inputs[index] == 1) {
        return;
    }

    nnue->inputs[index] = 1;

    // Update eval
    for (int i = 0; i < INNER_LAYER_COUNT; ++i) {
        nnue->accumulators[i] += nnue->weights_1[i][index];
    }

    nnue->eval = 0;

    for (int i = 0; i < INNER_LAYER_COUNT; ++i) {
        nnue->eval += ReLU(nnue->accumulators[i]) * nnue->weights_2[i];
    }
}

void resetNNUEInput(NNUE* nnue, int index) {
    if (!nnue->inputs[index]) {
        return;
    }

    nnue->inputs[index] = 0;

    counter--;

    // Update eval
    for (int i = 0; i < INNER_LAYER_COUNT; ++i) {
        nnue->accumulators[i] -= nnue->weights_1[i][index];
    }

    nnue->eval = 0;

    for (int i = 0; i < INNER_LAYER_COUNT; ++i) {
        nnue->eval += ReLU(nnue->accumulators[i]) * nnue->weights_2[i];
    }
}

void resetNNUE(NNUE* nnue) {
    for (int i = 0; i < INPUTS_COUNT; ++i) {
        nnue->inputs[i] = 0;
    }

    for (int i = 0; i < INNER_LAYER_COUNT; ++i) {
        nnue->accumulators[i] = 0;
    }

    nnue->eval = 0;
}


void modifyNnue(NNUE* nnue, Board* board, int color, int piece) {
    for(int sq = 0; sq < 64; ++sq) {
        int weight = isExists(board, color, piece, sq);
        if (weight) {
            setNNUEInput(nnue, getInputIndexOf(color, piece, sq));
        } else {
            resetNNUEInput(nnue, getInputIndexOf(color, piece, sq));
        }
    }
}

void loadNNUEWeights() {
    FILE* file = fopen("./fc1.weights.csv", "r");
    if (file == NULL) {
        perror("Unable to open file");
        exit(1);
    }

    for (int i = 0; i < INNER_LAYER_COUNT; i++) {
        for (int j = 0; j < INPUTS_COUNT; j++) {
            if (fscanf(file, "%lf,", &nnue->weights_1[i][j]) != 1) {
                perror("Error reading file");
                fclose(file);
                exit(1);
            }
        }
    }
    fclose(file);

    file = fopen("./fc2.weights.csv", "r");

    if (file == NULL) {
        perror("Unable to open file");
        exit(1);
    }

    for (int i = 0; i < INNER_LAYER_COUNT; i++) {
        if (fscanf(file, "%lf,", &nnue->weights_2[i]) != 1) {
            perror("Error reading file");
            fclose(file);
            exit(1);
        }
    }

    fclose(file);

    resetNNUE(nnue);
}

void initNNUEWeights() {
    // load file with weights
}

void initNNUEPosition(NNUE* nnue, Board* board) {
    resetNNUE(nnue);

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
}