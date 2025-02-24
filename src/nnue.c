#include <stdio.h>
#include <limits.h>
#ifdef __ARM_NEON
    #include <arm_neon.h>
#elif defined(__AVX2__)
    #include <immintrin.h>
#endif
#include "nnue.h"
#include "timemanager.h"
#include "search.h"
#include "weights.h"

int QA = 255;
int QB = 64;
int SCALE = 400;
double EXTRA_SCALE = 0.4;

int isExists(Board* board, int color, int piece, int sq) {
    return !!(board->pieces[piece] & board->colours[color] & bitboardCell(sq));
}

int getInputIndexOf(int color, int piece, int sq) {
    return color * 64 * 6 + (piece - 1) * 64 + sq;
}

double ReLU(double x) {
    return x > 0 ? x : 0;
}

#ifdef __ARM_NEON
void recalculateEval(NNUE* nnue, int color) {
    int32x4_t sum_vec_low = vdupq_n_s32(0);
    int32x4_t sum_vec_high = vdupq_n_s32(0);

    int shift = color == WHITE ? 0 : INNER_LAYER_COUNT;

    for (int i = 0; i < INNER_LAYER_COUNT; i += 8) {
        int32x4_t acc_vec_low = vld1q_s32(&nnue->accumulators[i]);
        int32x4_t acc_vec_high = vld1q_s32(&nnue->accumulators[i + 4]);

        int32x4_t w2_vec_low = vld1q_s32(&nnue->weights_2_quantized[shift + i]);
        int32x4_t w2_vec_high = vld1q_s32(&nnue->weights_2_quantized[shift + i + 4]);

        acc_vec_low = vmaxq_s32(acc_vec_low, vdupq_n_s32(0));
        acc_vec_high = vmaxq_s32(acc_vec_high, vdupq_n_s32(0));

        sum_vec_low = vmlaq_s32(sum_vec_low, acc_vec_low, w2_vec_low);
        sum_vec_high = vmlaq_s32(sum_vec_high, acc_vec_high, w2_vec_high);
    }

    nnue->eval = sum_vec_low[0] + sum_vec_low[1] + sum_vec_low[2] + sum_vec_low[3] +
                     sum_vec_high[0] + sum_vec_high[1] + sum_vec_high[2] + sum_vec_high[3];


    // part 2


    sum_vec_low = vdupq_n_s32(0);
    sum_vec_high = vdupq_n_s32(0);

    shift = color == WHITE ? INNER_LAYER_COUNT : 0;

    for (int i = 0; i < INNER_LAYER_COUNT; i += 8) {
        int32x4_t acc_vec_low = vld1q_s32(&nnue->accumulators_perspective[i]);
        int32x4_t acc_vec_high = vld1q_s32(&nnue->accumulators_perspective[i + 4]);

        int32x4_t w2_vec_low = vld1q_s32(&nnue->weights_2_quantized[i + shift]);
        int32x4_t w2_vec_high = vld1q_s32(&nnue->weights_2_quantized[i + shift + 4]);

        acc_vec_low = vmaxq_s32(acc_vec_low, vdupq_n_s32(0));
        acc_vec_high = vmaxq_s32(acc_vec_high, vdupq_n_s32(0));

        sum_vec_low = vmlaq_s32(sum_vec_low, acc_vec_low, w2_vec_low);
        sum_vec_high = vmlaq_s32(sum_vec_high, acc_vec_high, w2_vec_high);
    }

    int32_t result = sum_vec_low[0] + sum_vec_low[1] + sum_vec_low[2] + sum_vec_low[3] +
                     sum_vec_high[0] + sum_vec_high[1] + sum_vec_high[2] + sum_vec_high[3];

    nnue->eval += result;
    nnue->eval = nnue->eval / QA * SCALE / QB * EXTRA_SCALE;
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


#elif defined(__AVX2__)
static inline int horizontal_sum(__m256i vec) {
    __m128i low  = _mm256_castsi256_si128(vec);
    __m128i high = _mm256_extracti128_si256(vec, 1);
    __m128i sum128 = _mm_add_epi32(low, high);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    return _mm_cvtsi128_si32(sum128);
}

void recalculateEval(NNUE* nnue, int color) {
    __m256i sum_vec = _mm256_setzero_si256();
    int shift = (color == WHITE) ? 0 : INNER_LAYER_COUNT;

    for (int i = 0; i < INNER_LAYER_COUNT; i += 8) {
        __m256i acc = _mm256_loadu_si256((__m256i*)&nnue->accumulators[i]);
        __m256i w2  = _mm256_loadu_si256((__m256i*)&nnue->weights_2_quantized[shift + i]);
        acc = _mm256_max_epi32(acc, _mm256_setzero_si256());
        __m256i prod = _mm256_mullo_epi32(acc, w2);
        sum_vec = _mm256_add_epi32(sum_vec, prod);
    }
    int part1 = horizontal_sum(sum_vec);

    sum_vec = _mm256_setzero_si256();
    shift = (color == WHITE) ? INNER_LAYER_COUNT : 0;
    for (int i = 0; i < INNER_LAYER_COUNT; i += 8) {
        __m256i acc = _mm256_loadu_si256((__m256i*)&nnue->accumulators_perspective[i]);
        __m256i w2  = _mm256_loadu_si256((__m256i*)&nnue->weights_2_quantized[i + shift]);
        acc = _mm256_max_epi32(acc, _mm256_setzero_si256());
        __m256i prod = _mm256_mullo_epi32(acc, w2);
        sum_vec = _mm256_add_epi32(sum_vec, prod);
    }
    int part2 = horizontal_sum(sum_vec);

    int result = part1 + part2;

    nnue->eval = ((((result) / QA) * SCALE) / QB) * EXTRA_SCALE;
    nnue->eval = (color == WHITE) ? nnue->eval : -nnue->eval;
}

void setNNUEInput(S16* inputs, S32* accumulators, S32 (*weights)[INNER_LAYER_COUNT], int index) {
    if (inputs[index] == 1)
        return;

    inputs[index] = 1;
    for (int i = 0; i < INNER_LAYER_COUNT; i += 8) {
        __m256i acc = _mm256_loadu_si256((__m256i*)&accumulators[i]);
        __m256i w1  = _mm256_loadu_si256((__m256i*)&weights[index][i]);
        acc = _mm256_add_epi32(acc, w1);
        _mm256_storeu_si256((__m256i*)&accumulators[i], acc);
    }
}

void resetNNUEInput(S16* inputs, S32* accumulators, S32 (*weights)[INNER_LAYER_COUNT], int index) {
    if (inputs[index] == 0)
        return;

    inputs[index] = 0;
    for (int i = 0; i < INNER_LAYER_COUNT; i += 8) {
        __m256i acc = _mm256_loadu_si256((__m256i*)&accumulators[i]);
        __m256i w1  = _mm256_loadu_si256((__m256i*)&weights[index][i]);
        acc = _mm256_sub_epi32(acc, w1);
        _mm256_storeu_si256((__m256i*)&accumulators[i], acc);
    }
}

#endif

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

void loadInnerNNUEWeights() {
    int index = 0;
    for (int i = 0; i < INNER_LAYER_COUNT; i++) {
        for (int j = 0; j < INPUTS_COUNT; j++) {
            nnue->weights_1[i][j] = fc1_us[index];
            index++;
        }
    }

    for (int i = 0; i < INNER_LAYER_COUNT; i++) {
        for (int j = 0; j < INPUTS_COUNT; j++) {
            nnue->weights_1_quantized[j][i] = round(nnue->weights_1[i][j] * QA);
        }
    }


    index = 0;
    for (int i = 0; i < INNER_LAYER_COUNT; i++) {
        for (int j = 0; j < INPUTS_COUNT; j++) {
            nnue->weights_1_perspective[i][j] = fc1_them[index];
            index++;
        }
    }

    for (int i = 0; i < INNER_LAYER_COUNT; i++) {
        for (int j = 0; j < INPUTS_COUNT; j++) {
            nnue->weights_1_perspective_quantized[j][i] = round(nnue->weights_1_perspective[i][j] * QA);
        }
    }

    index = 0;
    for (int i = 0; i < 2 * INNER_LAYER_COUNT; i++) {
        nnue->weights_2[i] = fc2[index];
        index++;
    }

    for (int i = 0; i < 2 * INNER_LAYER_COUNT; i++) {
        nnue->weights_2_quantized[i] = round(nnue->weights_2[i] * QB);
    }

    resetNNUE(nnue);
}