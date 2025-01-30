#include <stdio.h>
#include <limits.h>
#include <arm_neon.h>
#include "nnue.h"
#include "timemanager.h"
#include "search.h"

int QA = 255;
int QB = 64;
double SCALE = 1.8;

int isExists(Board* board, int color, int piece, int sq) {
    return !!(board->pieces[piece] & board->colours[color] & bitboardCell(sq));
}

int getInputIndexOf(int color, int piece, int sq) {
  int rank = rankOf(sq);
  int file = fileOf(sq);
  int otherSq = square(7 - rank, file);

  return color * 64 * 6 + (piece - 1) * 64 + otherSq;
}

double ReLU(double x) {
    return x > 0 ? x : 0;
}

void recalculateEval(NNUE* nnue) {
  double eval = 0;

  for (int i = 0; i < INNER_LAYER_COUNT; i++) {
    double acc = nnue->accumulators[i] + nnue->biases_1[i];
    acc = ReLU(acc);
    eval += acc * acc * nnue->weights_2[i];
  }

    eval += nnue->biases_2;

    nnue->eval = eval;
}

//void recalculateEval(NNUE* nnue) {
//    int32x4_t sum_vec_low = vdupq_n_s32(0);
//    int32x4_t sum_vec_high = vdupq_n_s32(0);
//
//    for (int i = 0; i < INNER_LAYER_COUNT; i += 8) {
//        int32x4_t acc_vec_low = vld1q_s32(&nnue->accumulators[i]);
//        int32x4_t acc_vec_high = vld1q_s32(&nnue->accumulators[i + 4]);
//
//        int32x4_t w2_vec_low = vld1q_s32(&nnue->weights_2_quantized[i]);
//        int32x4_t w2_vec_high = vld1q_s32(&nnue->weights_2_quantized[i + 4]);
//        int32x4_t biases_1_vec_low = vld1q_s32(&nnue->biases_1_quantized[i]);
//        int32x4_t biases_1_vec_high = vld1q_s32(&nnue->biases_1_quantized[i + 4]);
//
//
//
//        acc_vec_low = vaddq_s32(acc_vec_low, biases_1_vec_low);
//
//        acc_vec_low = vmaxq_s32(acc_vec_low, vdupq_n_s32(0));
//        acc_vec_low = vmulq_s32(acc_vec_low, acc_vec_low);
//        // acc_vec_low = vminq_s32(acc_vec_low, vdupq_n_s32(QA));
//        acc_vec_low = vmulq_s32(acc_vec_low, acc_vec_low);
//        acc_vec_high = vaddq_s32(acc_vec_high, biases_1_vec_high);
//        acc_vec_high = vmaxq_s32(acc_vec_high, vdupq_n_s32(0));
//        // acc_vec_high = vminq_s32(acc_vec_high, vdupq_n_s32(QA));
//        acc_vec_high = vmulq_s32(acc_vec_high, acc_vec_high);
//
//        sum_vec_low = vmlaq_s32(sum_vec_low, acc_vec_low, w2_vec_low);
//        sum_vec_high = vmlaq_s32(sum_vec_high, acc_vec_high, w2_vec_high);
//    }
//
//    nnue->eval = sum_vec_low[0] + sum_vec_low[1] + sum_vec_low[2] + sum_vec_low[3] +
//                     sum_vec_high[0] + sum_vec_high[1] + sum_vec_high[2] + sum_vec_high[3];
//    nnue->eval += nnue->biases_2_quantized;
//    nnue->eval /= QA;
//    nnue->eval *= SCALE;
//
//    nnue->eval /= (QA * QB);
//
//    // printf("bias q: %d\n", nnue->biases_2_quantized);
//}

void setNNUEInput(NNUE* nnue, int index) {
    if (nnue->inputs[index] == 1) {
        return;
    }

    nnue->inputs[index] = 1;

    for(int i = 0; i < INNER_LAYER_COUNT; i++) {
        nnue->accumulators[i] += nnue->weights_1[i][index];
    }

//    for (int i = 0; i < INNER_LAYER_COUNT; i += 8) {
//        int32x4_t acc_vec_low = vld1q_s32(&nnue->accumulators[i]);
//        int32x4_t acc_vec_high = vld1q_s32(&nnue->accumulators[i + 4]);
//
//        int32x4_t w1_vec_low = vld1q_s32(&nnue->weights_1_quantized[index][i]);
//        int32x4_t w1_vec_high = vld1q_s32(&nnue->weights_1_quantized[index][i + 4]);
//
//        acc_vec_low = vaddq_s32(acc_vec_low, w1_vec_low);
//        acc_vec_high = vaddq_s32(acc_vec_high, w1_vec_high);
//
//        vst1q_s32(&nnue->accumulators[i], acc_vec_low);
//        vst1q_s32(&nnue->accumulators[i + 4], acc_vec_high);
//    }
}

void resetNNUEInput(NNUE* nnue, int index) {
    if (!nnue->inputs[index]) {
        return;
    }

    nnue->inputs[index] = 0;

    for(int i = 0; i < INNER_LAYER_COUNT; i++) {
        nnue->accumulators[i] -= nnue->weights_1[i][index];
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

void loadWeightsLayer(char* filename, double* weights, int rows, int cols) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Unable to open file");
        exit(1);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%lf,", &weights[i * cols + j]) != 1) {
                perror("Error reading file");
                fclose(file);
                exit(1);
            }
        }
    }

    fclose(file);
}


void loadNNUEWeights() {
    loadWeightsLayer("./fc1.weights.csv", nnue->weights_1[0], INNER_LAYER_COUNT, INPUTS_COUNT);
    loadWeightsLayer("./fc2.weights.csv", nnue->weights_2, INNER_LAYER_COUNT, 1);
    loadWeightsLayer("./fc1.biases.csv", nnue->biases_1, INNER_LAYER_COUNT, 1);
    loadWeightsLayer("./fc2.biases.csv", &nnue->biases_2, 1, 1);

    for (int i = 0; i < INNER_LAYER_COUNT; i++) {
        for (int j = 0; j < INPUTS_COUNT; j++) {
            nnue->weights_1_quantized[j][i] = round(nnue->weights_1[i][j] * QA);
        }
    }

    for (int i = 0; i < INNER_LAYER_COUNT; i++) {
        nnue->weights_2_quantized[i] = round(nnue->weights_2[i] * QB);
    }

    for (int i = 0; i < INNER_LAYER_COUNT; i++) {
        nnue->biases_1_quantized[i] = round(nnue->biases_1[i] * QA);
    }

    nnue->biases_2_quantized = round(nnue->biases_2 * QB);

    resetNNUE(nnue);
}

TimeManager createFixNodesTm(int nodes) {
    TimeManager tm = initTM();
    tm.nodes = nodes;
    tm.depth = 100;
    tm.searchType = FixedNodes;
    return tm;
}

int MAX_FEN_LENGTH = 1000;

void dataset_gen(Board* board, int from, int to, char* filename) {
    TimeManager tm = createFixNodesTm(20000);
    FILE *inputFile = fopen("training_data.txt", "r");
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