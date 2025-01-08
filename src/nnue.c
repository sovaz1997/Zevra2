#include <stdio.h>
#include <limits.h>
#include <arm_neon.h>
#include "nnue.h"
#include "timemanager.h"
#include "search.h"

int QA = 255;
int QB = 64;

int isExists(Board* board, int color, int piece, int sq) {
    return !!(board->pieces[piece] & board->colours[color] & bitboardCell(sq));
}

int getInputIndexOf(int color, int piece, int sq, int kingSq) {
    int pieceIndex = (piece - 1) * 2 + color;
    int res = sq + (pieceIndex + kingSq * 10) * 64;
    return res;
}

double ReLU(double x) {
    return x > 0 ? x : 0;
}

void recalculateEval(NNUE* nnue, int color) {
    int32x4_t sum_vec_low = vdupq_n_s32(0);
    int32x4_t sum_vec_high = vdupq_n_s32(0);

    int shift = 0;

    for (int i = 0; i < INNER_LAYER_COUNT; i += 8) {
        int32x4_t acc_vec_low = vld1q_s32(&nnue->accumulators[i]);
        int32x4_t acc_vec_high = vld1q_s32(&nnue->accumulators[i + 4]);

        int32x4_t w2_vec_low = vld1q_s32(&nnue->weights_2_quantized[shift + i]);
        int32x4_t w2_vec_high = vld1q_s32(&nnue->weights_2_quantized[shift + i + 4]);

        acc_vec_low = vmaxq_s32(acc_vec_low, vdupq_n_s32(0));
		acc_vec_low = vminq_s32(acc_vec_low, vdupq_n_s32(QA));
        acc_vec_high = vmaxq_s32(acc_vec_high, vdupq_n_s32(0));
        acc_vec_high = vminq_s32(acc_vec_high, vdupq_n_s32(QA));

        sum_vec_low = vmlaq_s32(sum_vec_low, acc_vec_low, w2_vec_low);
        sum_vec_high = vmlaq_s32(sum_vec_high, acc_vec_high, w2_vec_high);
    }

    nnue->eval = sum_vec_low[0] + sum_vec_low[1] + sum_vec_low[2] + sum_vec_low[3] +
                     sum_vec_high[0] + sum_vec_high[1] + sum_vec_high[2] + sum_vec_high[3];


    // part 2


    sum_vec_low = vdupq_n_s32(0);
    sum_vec_high = vdupq_n_s32(0);

    shift = INNER_LAYER_COUNT;

    for (int i = 0; i < INNER_LAYER_COUNT; i += 8) {
        int32x4_t acc_vec_low = vld1q_s32(&nnue->accumulators_perspective[i]);
        int32x4_t acc_vec_high = vld1q_s32(&nnue->accumulators_perspective[i + 4]);

        int32x4_t w2_vec_low = vld1q_s32(&nnue->weights_2_quantized[i + shift]);
        int32x4_t w2_vec_high = vld1q_s32(&nnue->weights_2_quantized[i + shift + 4]);

        acc_vec_low = vmaxq_s32(acc_vec_low, vdupq_n_s32(0));
		acc_vec_low = vminq_s32(acc_vec_low, vdupq_n_s32(QA));
        acc_vec_high = vmaxq_s32(acc_vec_high, vdupq_n_s32(0));
        acc_vec_high = vminq_s32(acc_vec_high, vdupq_n_s32(QA));

        sum_vec_low = vmlaq_s32(sum_vec_low, acc_vec_low, w2_vec_low);
        sum_vec_high = vmlaq_s32(sum_vec_high, acc_vec_high, w2_vec_high);
    }

    int32_t result = sum_vec_low[0] + sum_vec_low[1] + sum_vec_low[2] + sum_vec_low[3] +
                     sum_vec_high[0] + sum_vec_high[1] + sum_vec_high[2] + sum_vec_high[3];

    nnue->eval += result;
    nnue->eval /= (QA * QB);
    nnue->eval = color == WHITE ? nnue->eval : -nnue->eval;
}

void setNNUEInput(S16* inputs, S32* accumulators, S32 (*weights)[INNER_LAYER_COUNT], int index) {
    if (inputs[index] == 1) {
        return;
    }

    inputs[index] = 1;

    for (int i = 0; i < INNER_LAYER_COUNT; i += 8) {
        int32x4_t acc_vec_low = vld1q_s32(&accumulators[i]);
        int32x4_t acc_vec_high = vld1q_s32(&accumulators[i + 4]);

        int32x4_t w1_vec_low = vld1q_s32(&weights[index][i]);
        int32x4_t w1_vec_high = vld1q_s32(&weights[index][i + 4]);

        acc_vec_low = vaddq_s32(acc_vec_low, w1_vec_low);
        acc_vec_high = vaddq_s32(acc_vec_high, w1_vec_high);

        vst1q_s32(&accumulators[i], acc_vec_low);
        vst1q_s32(&accumulators[i + 4], acc_vec_high);
    }
}

void resetNNUEInput(S16* inputs, S32* accumulators, S32 (*weights)[INNER_LAYER_COUNT], int index) {
  	if (inputs[index] == 0) {
        return;
    }

    inputs[index] = 0;

    for (int i = 0; i < INNER_LAYER_COUNT; i += 8) {
        int32x4_t acc_vec_low = vld1q_s32(&accumulators[i]);
        int32x4_t acc_vec_high = vld1q_s32(&accumulators[i + 4]);

        int32x4_t w1_vec_low = vld1q_s32(&weights[index][i]);
        int32x4_t w1_vec_high = vld1q_s32(&weights[index][i + 4]);

        acc_vec_low = vsubq_s32(acc_vec_low, w1_vec_low);
        acc_vec_high = vsubq_s32(acc_vec_high, w1_vec_high);

        vst1q_s32(&accumulators[i], acc_vec_low);
        vst1q_s32(&accumulators[i + 4], acc_vec_high);
    }
}

void setDirectNNUEInput(NNUE* nnue, int index) {
  setNNUEInput(nnue->inputs, nnue->accumulators, nnue->weights_1_quantized, index);
}

void resetDirectNNUEInput(NNUE* nnue, int index) {
  resetNNUEInput(nnue->inputs, nnue->accumulators, nnue->weights_1_quantized, index);
}

void setPerspectiveNNUEInput(NNUE* nnue, int index) {
  setNNUEInput(nnue->inputs_perspective, nnue->accumulators_perspective, nnue->weights_1_perspective_quantized, index);
}

void resetPerspectiveNNUEInput(NNUE* nnue, int index) {
  resetNNUEInput(nnue->inputs_perspective, nnue->accumulators_perspective, nnue->weights_1_perspective_quantized, index);
}

void resetNNUE(NNUE* nnue) {
    for (int i = 0; i < INPUTS_COUNT; ++i) {
        nnue->inputs[i] = 0;
    }

    for (int i = 0; i < INPUTS_COUNT; ++i) {
        nnue->inputs_perspective[i] = 0;
    }

    for (int i = 0; i < INNER_LAYER_COUNT; ++i) {
        nnue->accumulators[i] = 0;
    }

    for (int i = 0; i < INNER_LAYER_COUNT; ++i) {
        nnue->accumulators_perspective[i] = 0;
    }

    nnue->eval = 0;
}


//void modifyNnue(NNUE* nnue, Board* board, int color, int piece) {
//    for(int sq = 0; sq < 64; ++sq) {
//        int weight = isExists(board, color, piece, sq);
//        if (weight) {
//            setNNUEInput(nnue, getInputIndexOf(color, piece, sq));
//        } else {
//            resetNNUEInput(nnue, getInputIndexOf(color, piece, sq));
//        }
//    }
//}

void loadNNUEWeights() {
    FILE* file = fopen("./fc1_us.weights.csv", "r");
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

    for (int i = 0; i < INNER_LAYER_COUNT; i++) {
        for (int j = 0; j < INPUTS_COUNT; j++) {
            nnue->weights_1_quantized[j][i] = round(nnue->weights_1[i][j] * QA);
        }
    }

    fclose(file);

    file = fopen("./fc1_them.weights.csv", "r");
    if (file == NULL) {
        perror("Unable to open file");
        exit(1);
    }

    for (int i = 0; i < INNER_LAYER_COUNT; i++) {
        for (int j = 0; j < INPUTS_COUNT; j++) {
            if (fscanf(file, "%lf,", &nnue->weights_1_perspective[i][j]) != 1) {
                perror("Error reading file");
                fclose(file);
                exit(1);
            }
        }
    }

    for (int i = 0; i < INNER_LAYER_COUNT; i++) {
        for (int j = 0; j < INPUTS_COUNT; j++) {
            nnue->weights_1_perspective_quantized[j][i] = round(nnue->weights_1_perspective[i][j] * QA);
        }
    }

    fclose(file);

    file = fopen("./fc2.weights.csv", "r");

    if (file == NULL) {
        perror("Unable to open file");
        exit(1);
    }

    for (int i = 0; i < 2 * INNER_LAYER_COUNT; i++) {
        if (fscanf(file, "%lf,", &nnue->weights_2[i]) != 1) {
            perror("Error reading file");
            fclose(file);
            exit(1);
        }
    }

    for (int i = 0; i < 2 * INNER_LAYER_COUNT; i++) {
        nnue->weights_2_quantized[i] = round(nnue->weights_2[i] * QB);
    }

    fclose(file);

    resetNNUE(nnue);
}
//
//void initNNUEPosition(NNUE* nnue, Board* board) {
//    resetNNUE(nnue);
//
//    modifyNnue(nnue, board, WHITE, PAWN);
//    modifyNnue(nnue, board, BLACK, PAWN);
//    modifyNnue(nnue, board, WHITE, KNIGHT);
//    modifyNnue(nnue, board, BLACK, KNIGHT);
//    modifyNnue(nnue, board, WHITE, BISHOP);
//    modifyNnue(nnue, board, BLACK, BISHOP);
//    modifyNnue(nnue, board, WHITE, ROOK);
//    modifyNnue(nnue, board, BLACK, ROOK);
//    modifyNnue(nnue, board, WHITE, QUEEN);
//    modifyNnue(nnue, board, BLACK, QUEEN);
//    modifyNnue(nnue, board, WHITE, KING);
//    modifyNnue(nnue, board, BLACK, KING);
//}

TimeManager createFixNodesTm(int nodes) {
    TimeManager tm = initTM();
    tm.nodes = nodes;
    tm.depth = 100;
    tm.searchType = FixedNodes;
    return tm;
}

int MAX_FEN_LENGTH = 1000;

void dataset_gen(Board* board, int from, int to, char* filename) {
    TimeManager tm = createFixNodesTm(10000);
    FILE *inputFile = fopen("ccrl_positions.txt", "r");
    FILE *outputFile = fopen(filename, "w");

    if (inputFile == NULL || outputFile == NULL) {
        perror("Ошибка открытия файла");
        exit(1);
    }

    char fen[MAX_FEN_LENGTH];
    int lineNumber = 0;

    while (fgets(fen, MAX_FEN_LENGTH, inputFile) != NULL) {
        lineNumber++;

        if (lineNumber < from) {
            continue;
        }

        if (lineNumber >= to) {
            fclose(inputFile);
            fclose(outputFile);
            exit(0);
            return;
        }

        char *newline = strchr(fen, '\n');
        if (newline) {
            *newline = '\0';
        }

        // Обрабатываем FEN-позицию
        printf("Позиция #%d: %s\n", lineNumber, fen);

        setFen(board, fen);
        SearchInfo info = iterativeDeeping(board, tm);
        int absoluteEval = board->color == WHITE ? info.eval : -info.eval;
        int staticEval = fullEval(board);
        int absoluteStaticEval = board->color == WHITE ? staticEval : -staticEval;
        int qEval = quiesceSearch(board, &info, -MATE_SCORE, MATE_SCORE, 0);
        int absoluteQEval = board->color == WHITE ? qEval : -qEval;

        if (
            abs(absoluteStaticEval - absoluteEval) > 70
            || abs(absoluteStaticEval - absoluteQEval) > 60
        ) {
            continue;
        }

        fprintf(outputFile, "%s,%d\n", fen, absoluteEval);
    }

    fclose(inputFile);
    fclose(outputFile);
    exit(0);
}