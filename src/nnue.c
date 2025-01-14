#include <stdio.h>
#include <limits.h>
#include <arm_neon.h>
#include "nnue.h"
#include "timemanager.h"
#include "search.h"

int QA = 255;
int QB = 64;
int SCALE = 400;

int isExists(Board* board, int color, int piece, int sq) {
    return !!(board->pieces[piece] & board->colours[color] & bitboardCell(sq));
}

int getInputIndexOf(int color, int piece, int sq) {
    return color * 64 * 6 + (piece - 1) * 64 + sq;
}

double ReLU(double x) {
    return x > 0 ? x : 0;
}

void multiply(
    S32* vec1,
    S32* accumulators,
    size_t size_vec1,
    size_t size_vec2,
    S32* weights,
    S32* biases
) {
    size_t j;

    int isLast = size_vec2 == 1;

    for (j = 0; j + 3 < size_vec2; j += 4) {
        int32x4_t sum_vec = vdupq_n_s32(0);
        size_t i = 0;

        for (; i + 3 < size_vec1; i += 4) {
            int32x4_t vdup0 = vdupq_n_s32(vec1[i + 0]);
            int32x4_t w0 = vld1q_s32(&weights[(i + 0) * size_vec2 + j]);
            sum_vec = vmlaq_s32(sum_vec, vdup0, w0);

            int32x4_t vdup1 = vdupq_n_s32(vec1[i + 1]);
            int32x4_t w1 = vld1q_s32(&weights[(i + 1) * size_vec2 + j]);
            sum_vec = vmlaq_s32(sum_vec, vdup1, w1);


            int32x4_t vdup2 = vdupq_n_s32(vec1[i + 2]);
            int32x4_t w2 = vld1q_s32(&weights[(i + 2) * size_vec2 + j]);
            sum_vec = vmlaq_s32(sum_vec, vdup2, w2);

            int32x4_t vdup3 = vdupq_n_s32(vec1[i + 3]);
            int32x4_t w3 = vld1q_s32(&weights[(i + 3) * size_vec2 + j]);
            sum_vec = vmlaq_s32(sum_vec, vdup3, w3);
        }

        for (; i < size_vec1; i++) {
            int32x4_t vdup = vdupq_n_s32(vec1[i]);
            int32x4_t w_vec = vld1q_s32(&weights[i * size_vec2 + j]);
            sum_vec = vmlaq_s32(sum_vec, vdup, w_vec);
        }

        vst1q_s32(&accumulators[j], sum_vec);
    }

    for (; j < size_vec2; j++) {
        S32 sum = 0;
        for (size_t i = 0; i < size_vec1; i++) {
            sum += vec1[i] * weights[i * size_vec2 + j];
        }
        accumulators[j] = sum;
    }
}

void recalculateEval(NNUE* nnue) {
    static S32 layer1_out[INNER_LAYER_COUNT];

     for (int i = 0; i < INNER_LAYER_COUNT; i += 8) {
         int32x4_t acc_low  = vld1q_s32(&nnue->accumulators[i]);
         int32x4_t acc_high = vld1q_s32(&nnue->accumulators[i + 4]);

         acc_low  = vmaxq_s32(acc_low,  vdupq_n_s32(0));
         acc_high = vmaxq_s32(acc_high, vdupq_n_s32(0));

         vst1q_s32(&layer1_out[i],     acc_low);
         vst1q_s32(&layer1_out[i + 4], acc_high);
    }

    for (int i = 0; i < INNER_LAYER_COUNT; i += 4) {
        int32x4_t v = vld1q_s32(&layer1_out[i]);
        v = vshrq_n_s32(v, 8);
        vst1q_s32(&layer1_out[i], v);
    }

    static S32 layer2_out[SECOND_INNER_LAYER_COUNT];

    multiply(
        layer1_out,
        layer2_out,
        INNER_LAYER_COUNT,
        SECOND_INNER_LAYER_COUNT,
        nnue->weights_2_quantized[0],
        nnue->biases_2_quantized
    );

    // relu
    int32x4_t zero = vdupq_n_s32(0);
    for (int i = 0; i < SECOND_INNER_LAYER_COUNT; i += 4) {
        int32x4_t v = vld1q_s32(&layer2_out[i]);
        int32x4_t result = vmaxq_s32(v, zero);
        vst1q_s32(&layer2_out[i], result);
    }

    S32 final_out[1];
    multiply(
        layer2_out,
        final_out,
        SECOND_INNER_LAYER_COUNT,
        1,
        nnue->weights_3_quantized,
        &nnue->biases_3_quantized
    );


    nnue->eval = final_out[0];

    nnue->eval *= SCALE;
    nnue->eval /= (QA * QB);
}

void setNNUEInput(NNUE* nnue, int index) {
    if (nnue->inputs[index] == 1) {
        return;
    }

    nnue->inputs[index] = 1;

    for (int i = 0; i < INNER_LAYER_COUNT; i += 8) {
        int32x4_t acc_vec_low = vld1q_s32(&nnue->accumulators[i]);
        int32x4_t acc_vec_high = vld1q_s32(&nnue->accumulators[i + 4]);

        int32x4_t w1_vec_low = vld1q_s32(&nnue->weights_1_quantized[index][i]);
        int32x4_t w1_vec_high = vld1q_s32(&nnue->weights_1_quantized[index][i + 4]);

        acc_vec_low = vaddq_s32(acc_vec_low, w1_vec_low);
        acc_vec_high = vaddq_s32(acc_vec_high, w1_vec_high);

        vst1q_s32(&nnue->accumulators[i], acc_vec_low);
        vst1q_s32(&nnue->accumulators[i + 4], acc_vec_high);
    }
}

void resetNNUEInput(NNUE* nnue, int index) {
    if (!nnue->inputs[index]) {
        return;
    }

    nnue->inputs[index] = 0;

    for (int i = 0; i < INNER_LAYER_COUNT; i += 8) {
        int32x4_t acc_vec_low = vld1q_s32(&nnue->accumulators[i]);
        int32x4_t acc_vec_high = vld1q_s32(&nnue->accumulators[i + 4]);

        int32x4_t w1_vec_low = vld1q_s32(&nnue->weights_1_quantized[index][i]);
        int32x4_t w1_vec_high = vld1q_s32(&nnue->weights_1_quantized[index][i + 4]);

        acc_vec_low = vsubq_s32(acc_vec_low, w1_vec_low);
        acc_vec_high = vsubq_s32(acc_vec_high, w1_vec_high);

        vst1q_s32(&nnue->accumulators[i], acc_vec_low);
        vst1q_s32(&nnue->accumulators[i + 4], acc_vec_high);
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

void loadQuantizedLayer(char* filename, double* weights, int size) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Unable to open file");
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%lf,", &weights[i]) != 1) {
            perror("Error reading file");
            fclose(file);
            exit(1);
        }
    }

    fclose(file);
}

void loadNNUEWeights() {
    loadWeightsLayer("./fc1.weights.csv", nnue->weights_1[0], INNER_LAYER_COUNT, INPUTS_COUNT);
    loadWeightsLayer("./fc2.weights.csv", nnue->weights_2[0], SECOND_INNER_LAYER_COUNT, INNER_LAYER_COUNT);
    loadWeightsLayer("./fc3.weights.csv", nnue->weights_3, SECOND_INNER_LAYER_COUNT, 1);

    for (int i = 0; i < INNER_LAYER_COUNT; i++) {
        for (int j = 0; j < INPUTS_COUNT; j++) {
            nnue->weights_1_quantized[j][i] = round(nnue->weights_1[i][j] * QA);
        }
    }

    for(int i = 0; i < SECOND_INNER_LAYER_COUNT; i++) {
        for(int j = 0; j < INNER_LAYER_COUNT; j++) {
            nnue->weights_2_quantized[j][i] = round(nnue->weights_2[i][j] * QA);
        }
    }

    for (int i = 0; i < SECOND_INNER_LAYER_COUNT; i++) {
        nnue->weights_3_quantized[i] = round(nnue->weights_3[i] * QB);
    }

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
    TimeManager tm = createFixNodesTm(5000);
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

//        if (
//            abs(absoluteStaticEval - absoluteEval) > 70
//            || abs(absoluteStaticEval - absoluteQEval) > 60
//        ) {
//            continue;
//        }

        fprintf(outputFile, "%s,%d\n", fen, absoluteEval);
    }

    fclose(inputFile);
    fclose(outputFile);
    exit(0);
}