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

void recalculateEval(NNUE* nnue) {
    int32x4_t sum_vec_low = vdupq_n_s32(0);
    int32x4_t sum_vec_high = vdupq_n_s32(0);

    for (int i = 0; i < INNER_LAYER_COUNT; i += 8) {
        int32x4_t acc_vec_low = vld1q_s32(&nnue->accumulators[i]);
        int32x4_t acc_vec_high = vld1q_s32(&nnue->accumulators[i + 4]);

        int32x4_t w2_vec_low = vld1q_s32(&nnue->weights_2_quantized[i]);
        int32x4_t w2_vec_high = vld1q_s32(&nnue->weights_2_quantized[i + 4]);
//		printf("weights  %d %d %d %d\n", nnue->weights_2_quantized[i], nnue->weights_2_quantized[i + 1], nnue->weights_2_quantized[i + 2], nnue->weights_2_quantized[i + 3]);
//        printf("accumulators  %d %d %d %d %d %d %d %d\n", nnue->accumulators[i], nnue->accumulators[i + 1], nnue->accumulators[i + 2], nnue->accumulators[i + 3],
//                       nnue->accumulators[i + 4], nnue->accumulators[i + 5], nnue->accumulators[i + 6], nnue->accumulators[i + 7]);


        acc_vec_low = vmaxq_s32(acc_vec_low, vdupq_n_s32(0));
        // acc_vec_low = vminq_s32(acc_vec_low, vdupq_n_s32(QA));
        acc_vec_high = vmaxq_s32(acc_vec_high, vdupq_n_s32(0));
        // acc_vec_high = vminq_s32(acc_vec_high, vdupq_n_s32(QA));

        sum_vec_low = vmlaq_s32(sum_vec_low, acc_vec_low, w2_vec_low);
        sum_vec_high = vmlaq_s32(sum_vec_high, acc_vec_high, w2_vec_high);
    }

    int32_t result = sum_vec_low[0] + sum_vec_low[1] + sum_vec_low[2] + sum_vec_low[3] +
                     sum_vec_high[0] + sum_vec_high[1] + sum_vec_high[2] + sum_vec_high[3];
    nnue->eval = result / QA * SCALE / QB;
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

    for (int i = 0; i < INNER_LAYER_COUNT; i++) {
        for (int j = 0; j < INPUTS_COUNT; j++) {
            nnue->weights_1_quantized[j][i] = round(nnue->weights_1[i][j] * QA);
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

    for (int i = 0; i < INNER_LAYER_COUNT; i++) {
        nnue->weights_2_quantized[i] = round(nnue->weights_2[i] * QB);
    }

    fclose(file);

    resetNNUE(nnue);
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

TimeManager createFixNodesTm(int nodes) {
    TimeManager tm = initTM();
    tm.nodes = nodes;
    tm.depth = 100;
    tm.searchType = FixedNodes;
    return tm;
}

void genDataset(Board* board, int from, int to, char* filename) {
    TimeManager tm = createFixNodesTm(5000);
    FILE *inputFile = fopen("billion-dataset.fen", "r");
    FILE *outputFile = fopen(filename, "a");

    if (inputFile == NULL || outputFile == NULL) {
        perror("Ошибка открытия файла");
        exit(1);
    }

    char fen[MAX_FEN_LENGTH];
    int lineNumber = 0;
    U16 moveList[256];
    Undo undo;

    while (fgets(fen, MAX_FEN_LENGTH, inputFile) != NULL) {
        lineNumber++;

        if (lineNumber < from) {
            continue;
        }

        if (lineNumber > to) {
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

		U16* moveListPtr = moveList;
    	movegen(board, moveList);

        while (*moveListPtr) {
            makeMove(board, *moveListPtr, &undo);
            checkAndSavePositionToDataset(board, tm, outputFile);
            unmakeMove(board, *moveListPtr, &undo);
            ++moveListPtr;
        }
    }

    fclose(inputFile);
    fclose(outputFile);
    exit(0);
}

void checkAndSavePositionToDataset(Board* board, TimeManager tm, FILE* file) {
	SearchInfo info = iterativeDeeping(board, tm);
    int absoluteEval = board->color == WHITE ? info.eval : -info.eval;
    int staticEval = fullEval(board);
    int absoluteStaticEval = board->color == WHITE ? staticEval : -staticEval;
    int qEval = quiesceSearch(board, &info, -MATE_SCORE, MATE_SCORE, 0);
    int absoluteQEval = board->color == WHITE ? qEval : -qEval;

    if (abs(absoluteStaticEval - absoluteEval) <= 70
        && abs(absoluteStaticEval - absoluteQEval) <= 60
        && !inCheck(board, board->color)
        && !inCheck(board, !board->color)
        && !havePromotionPawn(board)) {
            getFen(board, fen);
            fprintf(file, "%s,%d\n", fen, absoluteEval);
    }
}