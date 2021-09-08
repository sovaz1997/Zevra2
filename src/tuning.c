#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <malloc.h>
#include <stdlib.h>

#include "tuning.h"
#include "search.h"

double K = 0.9166;
const int PARAMS_COUNT = 858;

struct TuningPosition {
    char fen[512];
    int movesToEnd;
    double result;
};


struct LinearEvalPosition {
    int paramsCount;
    LinearEvalParam* params;
};

struct LinearEvalParam {
    int number;
    double k;
};

LinearEvalPosition* linearEvalPositions;
double* linearEvals;
int* evalParams;

void makeTuning(Board *board) {
    loadPositions(board);

    evalParams = getValues();

    double E = fun(board);

    while(1) {
        int improved = 0;
        int iterations = 0;
        for (int i = 1; i < PARAMS_COUNT; i++) {
            incParam(evalParams, i);

            double newE = fun(board);

            printf("NewE: %.7f; E: %.7f, index: %d; value: %d\n", newE, E, i, evalParams[i]);
            // printf("NewE: %.7f; index: %d; value: %d\n", E, i, evalParams[i]);
            if (newE < E) {
                improved = 1;
                E = newE;
                printParams();
                iterations++;
                printf("NewE: %.7f; index: %d; value: %d\n", E, i, evalParams[i]);
            } else {
                decParam(evalParams, i);
                decParam(evalParams, i);

                newE = fun(board);

                if (newE < E) {
                    decParam(evalParams, i);
                    improved = 1;
                    E = newE;
                    printParams();
                    iterations++;
                    printf("NewE: %.7f; index: %d; value: %d\n", E, i, evalParams[i]);
                } else {
                    incParam(evalParams, i);
                }
            }
        }

        printf("Iterations: %d/%d\n", iterations, PARAMS_COUNT);

        if (!improved) {
            break;
        }
    }


//    while (1) {
//        int improved = 0;
//        int iterations = 0;
//        for (int i = 1; i < PARAMS_COUNT; i++) {
//            int tmpParam = curValues[i];
//            changeParam(i, curValues[i] + changeFactor);
//
//            double newE = fun(board);
//
//            printf("%d %.7f %.7f\n", i, E, newE);
//
//            if (newE < E) {
//                while (newE < E) {
//                    improved = 1;
//                    curValues[i] += changeFactor;
//                    E = newE;
//                    printParams();
//                    iterations++;
//                    printf("NewE: %.7f; index: %d; value: %d\n", E, i, curValues[i]);
//                    changeParam(i, curValues[i] + changeFactor);
//                    newE = fun(board);
//                }
//                changeParam(i, curValues[i] - changeFactor);
//                curValues[i] -= changeFactor;
//            } else {
//                changeParam(i, curValues[i] - changeFactor);
//
//                newE = fun(board);
//
//                if (newE < E) {
//                    while (newE < E) {
//                        improved = 1;
//                        curValues[i] -= changeFactor;
//                        E = newE;
//                        printParams();
//                        iterations++;
//                        printf("NewE: %.7f; index: %d; value: %d\n", E, i, curValues[i]);
//                        changeParam(i, curValues[i] - changeFactor);
//                        newE = fun(board);
//                    }
//                    changeParam(i, curValues[i] + changeFactor);
//                    curValues[i] += changeFactor;
//                } else {
//                    changeParam(i, tmpParam);
//                    curValues[i] = tmpParam;
//                }
//            }
//        }

//        printf("Iterations: %d/%d\n", iterations, PARAMS_COUNT);
//
//        if (!improved) {
//            break;
//        }
//    }

    for (int i = 0; i < PARAMS_COUNT; i++) {
        printf("%d ", evalParams[i]);
    }
}

int positionsCount = 0;
TuningPosition *positions;

void loadPositions(Board *board) {
    FILE *f = fopen("all-test-positions.txt", "r");

    SearchInfo searchInfo;
    TimeManager tm = createFixDepthTm(MAX_PLY - 1);
    resetSearchInfo(&searchInfo, tm);

    char buf[4096];
    char *estr;

    int quiets = 0;

    int N = 120000000;
    positions = malloc(sizeof(TuningPosition) * N);
    linearEvalPositions = malloc(sizeof(LinearEvalPosition) * N);
    linearEvals = malloc(sizeof(double ) * N);

    while (1) {
        if (positionsCount > 1000) {
            break;
        }

        estr = fgets(buf, sizeof(buf), f);

        if (!estr) {
            break;
        }

        char **res = str_split(estr, ',');

        char *fen = *res;
        int movesToEnd = atoi(*(res + 1));
        double result = atof(*(res + 2));


        setFen(board, fen);
        int eval = fullEval(board);
        int qEval = quiesceSearch(board, &searchInfo, -MATE_SCORE, MATE_SCORE, 0);


        if (abs(eval - qEval) < 50 && popcount(board->colours[WHITE] | board->colours[BLACK]) > 7) {
            ++positionsCount;

            calculateLinear(board, positionsCount - 1);

            linearEvals[positionsCount - 1] = getLinearEval(positionsCount - 1);

            if (positionsCount % 100 == 0) {
                printf("%d\n", positionsCount);
            }

            strcpy(positions[positionsCount - 1].fen, fen);
            if (positionsCount % 100000 == 0) {
                printf("Pos count: %d\n", positionsCount);
            }

            positions[positionsCount - 1].result = result;
            positions[positionsCount - 1].movesToEnd = movesToEnd;

            quiets++;
        }

        for (int i = 0; i < 3; i++) {
            free(res[i]);
        }

        free(res);
    }
    printf("Pos count: %d; quiets count: %d\n", positionsCount, quiets);

    fclose(f);

}

char **str_split(char *a_str, const char a_delim) {
    char **result = 0;
    size_t count = 0;
    char *tmp = a_str;
    char *last_comma = 0;
    char delim[2];
    delim[0] = a_delim;
    delim[1] = 0;

    while (*tmp) {
        if (a_delim == *tmp) {
            count++;
            last_comma = tmp;
        }
        tmp++;
    }

    count += last_comma < (a_str + strlen(a_str) - 1);

    count++;

    result = malloc(sizeof(char *) * count);

    if (result) {
        size_t idx = 0;
        char *token = strtok(a_str, delim);

        while (token) {
            assert(idx < count);
            *(result + idx++) = strdup(token);
            token = strtok(0, delim);
        }
        assert(idx == count - 1);
        *(result + idx) = 0;
    }

    return result;
}

double r(double eval) {
    return 1. / (1. + pow(10, -K * eval / 400.));
}

double fun(Board *board) {
    SearchInfo searchInfo;
    TimeManager tm = createFixDepthTm(MAX_PLY - 1);
    resetSearchInfo(&searchInfo, tm);

    int posCount = 0;
    const double fadingFactor = 40;

    double errorSums = 0;


    for (int i = 0; i < positionsCount; i++) {
        resetSearchInfo(&searchInfo, tm);

        char *fen = positions[i].fen;
        int movesToEnd = positions[i].movesToEnd;
        double result = positions[i].result;

        double fading = exp(-movesToEnd / fadingFactor);

        setFen(board, fen);

        // int eval = getLinearEval(i);// fullEval(board);
        double eval = linearEvals[i];

        /*if (board->color == BLACK) {
            eval = -eval;
        }*/

        double error = pow(r(eval) - result, 2) * fading;
        errorSums += error;

        ++posCount;
    }

    errorSums /= posCount;

    return errorSums;
}

void setValues(int *values, int stage) {
    int curIndex = 0;

    transfer(&values[curIndex], &PAWN_EV_MG, &curIndex, 1);
    transfer(&values[curIndex], &PAWN_EV_EG, &curIndex, 1);
    transfer(&values[curIndex], &KNIGHT_EV_MG, &curIndex, 1);
    transfer(&values[curIndex], &KNIGHT_EV_EG, &curIndex, 1);
    transfer(&values[curIndex], &BISHOP_EV_MG, &curIndex, 1);
    transfer(&values[curIndex], &BISHOP_EV_EG, &curIndex, 1);
    transfer(&values[curIndex], &ROOK_EV_MG, &curIndex, 1);
    transfer(&values[curIndex], &ROOK_EV_EG, &curIndex, 1);
    transfer(&values[curIndex], &QUEEN_EV_MG, &curIndex, 1);
    transfer(&values[curIndex], &QUEEN_EV_EG, &curIndex, 1);

    // PST
    transferPST(&values[curIndex], pawnPST, &curIndex);
    transferPST(&values[curIndex], knightPST, &curIndex);
    transferPST(&values[curIndex], bishopPST, &curIndex);
    transferPST(&values[curIndex], rookPST, &curIndex);
    transferPST(&values[curIndex], queenPST, &curIndex);
    transferPST(&values[curIndex], kingPST, &curIndex);
    transferPST(&values[curIndex], egPawnPST, &curIndex);
    transferPST(&values[curIndex], egKnightPST, &curIndex);
    transferPST(&values[curIndex], egBishopPST, &curIndex);
    transferPST(&values[curIndex], egRookPST, &curIndex);
    transferPST(&values[curIndex], egQueenPST, &curIndex);
    transferPST(&values[curIndex], egKingPST, &curIndex);

    // Mobility
    transfer(&values[curIndex], QueenMobility, &curIndex, 28);
    transfer(&values[curIndex], RookMobility, &curIndex, 15);
    transfer(&values[curIndex], BishopMobility, &curIndex, 14);
    transfer(&values[curIndex], KnightMobility, &curIndex, 8);


    transfer(&values[curIndex], PassedPawnBonus, &curIndex, 8);

    transfer(&values[curIndex], &DoublePawnsPenalty, &curIndex, 1);
    transfer(&values[curIndex], &IsolatedPawnPenalty, &curIndex, 1);
    transfer(&values[curIndex], &DoubleBishopsBonusMG, &curIndex, 1);
    transfer(&values[curIndex], &DoubleBishopsBonusEG, &curIndex, 1);
    transfer(&values[curIndex], &KingDangerFactor, &curIndex, 1);
    transfer(&values[curIndex], &RookOnOpenFileBonus, &curIndex, 1);
    transfer(&values[curIndex], &RookOnPartOpenFileBonus, &curIndex, 1);


    // re-init due to dependent eval
    initDependencyStagedEval(stage);
}

int *transfer(int *from, int *to, int *curIndex, int length) {
    for (int i = 0; i < length; i++) {
        to[i] = *(from + i);
        (*curIndex)++;
    }
}

int *transferPST(int *from, int *to, int *curIndex) {
    transfer(from, to, curIndex, 64);
}

int *getValues() {
    int *res = (int *) malloc(sizeof(int) * PARAMS_COUNT);

    int curIndex = 0;

    transfer(&PAWN_EV_MG, &res[curIndex], &curIndex, 1);
    transfer(&PAWN_EV_EG, &res[curIndex], &curIndex, 1);
    transfer(&KNIGHT_EV_MG, &res[curIndex], &curIndex, 1);
    transfer(&KNIGHT_EV_EG, &res[curIndex], &curIndex, 1);
    transfer(&BISHOP_EV_MG, &res[curIndex], &curIndex, 1);
    transfer(&BISHOP_EV_EG, &res[curIndex], &curIndex, 1);
    transfer(&ROOK_EV_MG, &res[curIndex], &curIndex, 1);
    transfer(&ROOK_EV_EG, &res[curIndex], &curIndex, 1);
    transfer(&QUEEN_EV_MG, &res[curIndex], &curIndex, 1);
    transfer(&QUEEN_EV_EG, &res[curIndex], &curIndex, 1);


    // PST
    transferPST(pawnPST, &res[curIndex], &curIndex);
    transferPST(knightPST, &res[curIndex], &curIndex);
    transferPST(bishopPST, &res[curIndex], &curIndex);
    transferPST(rookPST, &res[curIndex], &curIndex);
    transferPST(queenPST, &res[curIndex], &curIndex);
    transferPST(kingPST, &res[curIndex], &curIndex);
    transferPST(egPawnPST, &res[curIndex], &curIndex);
    transferPST(egKnightPST, &res[curIndex], &curIndex);
    transferPST(egBishopPST, &res[curIndex], &curIndex);
    transferPST(egRookPST, &res[curIndex], &curIndex);
    transferPST(egQueenPST, &res[curIndex], &curIndex);
    transferPST(egKingPST, &res[curIndex], &curIndex);

    // Mobility
    transfer(QueenMobility, &res[curIndex], &curIndex, 28);
    transfer(RookMobility, &res[curIndex], &curIndex, 15);
    transfer(BishopMobility, &res[curIndex], &curIndex, 14);
    transfer(KnightMobility, &res[curIndex], &curIndex, 8);

    transfer(PassedPawnBonus, &res[curIndex], &curIndex, 8);

    transfer(&DoublePawnsPenalty, &res[curIndex], &curIndex, 1);
    transfer(&IsolatedPawnPenalty, &res[curIndex], &curIndex, 1);
    transfer(&DoubleBishopsBonusMG, &res[curIndex], &curIndex, 1);
    transfer(&DoubleBishopsBonusEG, &res[curIndex], &curIndex, 1);
    transfer(&KingDangerFactor, &res[curIndex], &curIndex, 1);
    transfer(&RookOnOpenFileBonus, &res[curIndex], &curIndex, 1);
    transfer(&RookOnPartOpenFileBonus, &res[curIndex], &curIndex, 1);

    return res;
}

void changeParam(int n, int value) {
    int *params = getValues();
    *(params + n) = value;
    setValues(params, 1); // TODO: исп. по-другому
    free(params);
}

void incParam(int* arr, int n) {
    (*(arr + n))++;
    for (int i = 0; i < positionsCount; i++) {
        for (int j = 0; j < linearEvalPositions[i].paramsCount; j++) {
            if (n == linearEvalPositions[i].params[j].number) {
                linearEvals[i] += linearEvalPositions[i].params[j].k;
                break;
            }
        }
    }
}

void decParam(int* arr, int n) {
    (*(arr + n))--;
    for (int i = 0; i < positionsCount; i++) {
        for (int j = 0; j < linearEvalPositions[i].paramsCount; j++) {
            if (n == linearEvalPositions[i].params[j].number) {
                linearEvals[i] -= linearEvalPositions[i].params[j].k;
                break;
            }
        }
    }
}

void printParams() {
    int *params = evalParams; //getValues();

    int curIndex = 0;

    FILE *f;
    char name[] = "weights.txt";

    if ((f = fopen(name, "w")) == NULL) {
        printf("Не удалось открыть файл");
        return;
    }


    printArray("PAWN_EV_MG", &params[curIndex], &curIndex, 1, f);
    printArray("PAWN_EG", &params[curIndex], &curIndex, 1, f);
    printArray("KNIGHT_EV_MG", &params[curIndex], &curIndex, 1, f);
    printArray("KNIGHT_EV_EG", &params[curIndex], &curIndex, 1, f);
    printArray("BISHOP_EV_MG", &params[curIndex], &curIndex, 1, f);
    printArray("BISHOP_EV_EG", &params[curIndex], &curIndex, 1, f);
    printArray("ROOK_EV_MG", &params[curIndex], &curIndex, 1, f);
    printArray("ROOK_EV_EG", &params[curIndex], &curIndex, 1, f);
    printArray("QUEEN_EV_MG", &params[curIndex], &curIndex, 1, f);
    printArray("QUEEN_EV_EG", &params[curIndex], &curIndex, 1, f);

    printPST("pawnPST", &params[curIndex], &curIndex, f);
    printPST("knightPST", &params[curIndex], &curIndex, f);
    printPST("bishopPST", &params[curIndex], &curIndex, f);
    printPST("rookPST", &params[curIndex], &curIndex, f);
    printPST("queenPST", &params[curIndex], &curIndex, f);
    printPST("kingPST", &params[curIndex], &curIndex, f);
    printPST("egPawnPST", &params[curIndex], &curIndex, f);
    printPST("egKnightPST", &params[curIndex], &curIndex, f);
    printPST("egBishopPST", &params[curIndex], &curIndex, f);
    printPST("egRookPST", &params[curIndex], &curIndex, f);
    printPST("egQueenPST", &params[curIndex], &curIndex, f);
    printPST("egKingPST", &params[curIndex], &curIndex, f);

    // Mobility
    printArray("QueenMobility", &params[curIndex], &curIndex, 28, f);
    printArray("RookMobility", &params[curIndex], &curIndex, 15, f);
    printArray("BishopMobility", &params[curIndex], &curIndex, 14, f);
    printArray("KnightMobility", &params[curIndex], &curIndex, 8, f);

    printArray("PassedPawnsBonus", &params[curIndex], &curIndex, 8, f);

    printArray("DoublePawnsPenalty", &params[curIndex], &curIndex, 1, f);
    printArray("IsolatedPawnPenalty", &params[curIndex], &curIndex, 1, f);
    printArray("DoubleBishopsBonusMG", &params[curIndex], &curIndex, 1, f);
    printArray("DoubleBishopsBonusEG", &params[curIndex], &curIndex, 1, f);
    printArray("KingDangerFactor", &params[curIndex], &curIndex, 1, f);
    printArray("RookOnOpenFileBonus", &params[curIndex], &curIndex, 1, f);
    printArray("RookOnPartOpenFileBonus", &params[curIndex], &curIndex, 1, f);

    fclose(f);
}

/**
 * Вычисление линейной функции оценки для определенной позиции
 * @param b
 * @return
 */
int *calculateLinear(Board *board, int positionNumber) {
    int* prevValues = getValues();

    linearEvalPositions[positionNumber].params = malloc(sizeof(LinearEvalParam) * PARAMS_COUNT);

    double *linearEval = malloc(PARAMS_COUNT * sizeof(double));
    int *values = malloc(PARAMS_COUNT * sizeof(int));
    memset(values, 0, PARAMS_COUNT * sizeof(int));

    int stage = stageGame(board);
    setValues(values, stage);

    double up = 1000;

    int evalParamsCount = 0;
    for (int i = 0; i < PARAMS_COUNT; i++) {
        values[i] = up;
        setValues(values, stage);

        int ev = fullEval(board);

        linearEval[i] = ev / up;
        values[i] = 0;

        if (ev != 0) {
            linearEvalPositions[positionNumber].params[evalParamsCount].number = i;
            linearEvalPositions[positionNumber].params[evalParamsCount].k = linearEval[i];
            evalParamsCount++;
        }
    }
    linearEvalPositions[positionNumber].params = realloc(
            linearEvalPositions[positionNumber].params,
            sizeof(LinearEvalParam) * evalParamsCount
    );

    linearEvalPositions[positionNumber].paramsCount = evalParamsCount;

    free(values);

    setValues(prevValues, stage);
    free(prevValues);
}

double getLinearEval(int positionNumber) {
    double eval = 0;
    int* evalSettings = getValues();
    for (int i = 0; i < linearEvalPositions[positionNumber].paramsCount; i++) {
        double k = linearEvalPositions[positionNumber].params[i].k;
        int evalParamIndex = linearEvalPositions[positionNumber].params[i].number;
        eval += evalSettings[evalParamIndex] * k;
    }
    free(evalSettings);
    return eval;
}

void printPST(char *name, int *pst, int *curIndex, FILE *f) {
    fprintf(f, "%s\n", name);
    for (int i = 0; i < 64; ++i, (*curIndex)++) {
        fprintf(f, "%d, ", pst[i]);
        if (i > 0 && (i + 1) % 8 == 0) {
            fprintf(f, "\n");
        }
    }
}

void printArray(char *name, int *arr, int *curIndex, int length, FILE *f) {
    if (length == 1) {
        fprintf(f, "%s: %d\n", name, arr[0]);
        (*curIndex)++;
        return;
    }

    fprintf(f, "%s\n", name);
    for (int i = 0; i < length; ++i, (*curIndex)++) {
        fprintf(f, "%d, ", arr[i]);
    }
    fprintf(f, "\n");
}