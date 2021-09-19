#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#include "tuning.h"
#include "search.h"

double K = 0.9166;
enum { PARAMS_COUNT = 925 };

struct TuningPosition {
    char fen[512];
    int movesToEnd;
    double result;
    int mul;
};

struct ParameterInfluencePositionList {
    int positionsCount;
    int* positionsList;
};

double **linearEvalPositions;
double *linearEvals;
int *evalParams;
ParameterInfluencePositionList* paramsInfluence;

void makeTuning(Board *board) {
    loadPositions(board);

    evalParams = getValues();

    double E = fun();

    FILE* regression = fopen("regression.txt", "w");

    int stage = 1;

    printParams("first-weights.txt", "first-linear-weights.txt");

    while(1) {
        int improved = 0;
        int iterations = 0;
        for (int i = 1; i < PARAMS_COUNT; i++) {
            incParam(evalParams, i, 1);

            double newE = fun();

            if (newE < E) {
                improved = 1;
                E = newE;
                iterations++;
            } else {
                incParam(evalParams, i, -2);

                newE = fun();

                if (newE < E) {
                    improved = 1;
                    E = newE;
                    iterations++;
                } else {
                    incParam(evalParams, i, 1);
                }
            }
        }

        char stageStr[10];
        itoa(stage, stageStr, 10);

        char linearFileName[256] = "";
        char fileName[256] = "";

        strcat(linearFileName, "linear-weights-");
        strcat(fileName, "weights-");

        strcat(linearFileName, stageStr);
        strcat(fileName, stageStr);

        strcat(linearFileName, ".txt");
        strcat(fileName, ".txt");

        printParams(fileName, linearFileName);

        printf("NewE: %.7f\n", E);
        printf("Iterations: %d/%d\n", iterations, PARAMS_COUNT);

        ++stage;
        fprintf(regression, "%f\n", E);

        if (!improved) {
            break;
        }
    }

    for (int i = 0; i < PARAMS_COUNT; i++) {
        printf("%d ", evalParams[i]);
    }

    fclose(regression);
}

int positionsCount = 0;
TuningPosition *positions;

void loadPositions(Board *board) {
    FILE *f = fopen("2000000.txt", "r");

    SearchInfo searchInfo;
    TimeManager tm = createFixDepthTm(MAX_PLY - 1);
    resetSearchInfo(&searchInfo, tm);

    char buf[4096];
    char *estr;

    int quiets = 0;

    int N = 12000000;
    positions = malloc(sizeof(TuningPosition) * N);
    linearEvalPositions = malloc(sizeof(double *) * N);
    linearEvals = malloc(sizeof(double) * N);
    paramsInfluence = malloc(sizeof(struct ParameterInfluencePositionList) * PARAMS_COUNT);

    while (1) {
        if (positionsCount >= 10000) {
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
            positions[positionsCount - 1].mul = board->color == WHITE ? 1 : -1;

            quiets++;
        }

        for (int i = 0; i < 3; i++) {
            free(res[i]);
        }

        free(res);
    }


    for (int i = 0; i < PARAMS_COUNT; ++i) {
        paramsInfluence[i].positionsList = malloc(sizeof(int) * 10000000);
        int currentEvalPositionsCount = 0;
        for (int j = 0; j < positionsCount; ++j) {
            if (linearEvalPositions[j][i] != 0) {
                paramsInfluence[i].positionsList[currentEvalPositionsCount] = j;
                currentEvalPositionsCount++;
            }
        }
        paramsInfluence[i].positionsCount = currentEvalPositionsCount;
        paramsInfluence[i].positionsList = realloc(paramsInfluence[i].positionsList, sizeof(int) * currentEvalPositionsCount);
        // printf("Param %d; Count: %d\n", i, sum);
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

double getErrorFromPosition(int posNumber) {

}

double fun() {
    int posCount = 0;
    const double fadingFactor = 40;
    double errorSums = 0;

    for (int i = 0; i < positionsCount; i++) {
        int movesToEnd = positions[i].movesToEnd;
        double fading = exp(-movesToEnd / fadingFactor);
        double eval = linearEvals[i] * positions[i].mul;

        double error = pow(r(eval) - positions[i].result, 2) * fading;
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
    transfer(&values[curIndex], QueenMobilityMG, &curIndex, QUEEN_MOBILITY_N);
    transfer(&values[curIndex], RookMobilityMG, &curIndex, ROOK_MOBILITY_N);
    transfer(&values[curIndex], BishopMobilityMG, &curIndex, BISHOP_MOBILITY_N);
    transfer(&values[curIndex], KnightMobilityMG, &curIndex, KNIGHT_MOBILITY_N);
    transfer(&values[curIndex], QueenMobilityEG, &curIndex, QUEEN_MOBILITY_N);
    transfer(&values[curIndex], RookMobilityEG, &curIndex, ROOK_MOBILITY_N);
    transfer(&values[curIndex], BishopMobilityEG, &curIndex, BISHOP_MOBILITY_N);
    transfer(&values[curIndex], KnightMobilityEG, &curIndex, KNIGHT_MOBILITY_N);


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
    transfer(QueenMobilityMG, &res[curIndex], &curIndex, QUEEN_MOBILITY_N);
    transfer(RookMobilityMG, &res[curIndex], &curIndex, ROOK_MOBILITY_N);
    transfer(BishopMobilityMG, &res[curIndex], &curIndex, BISHOP_MOBILITY_N);
    transfer(KnightMobilityMG, &res[curIndex], &curIndex, KNIGHT_MOBILITY_N);
    transfer(QueenMobilityEG, &res[curIndex], &curIndex, QUEEN_MOBILITY_N);
    transfer(RookMobilityEG, &res[curIndex], &curIndex, ROOK_MOBILITY_N);
    transfer(BishopMobilityEG, &res[curIndex], &curIndex, BISHOP_MOBILITY_N);
    transfer(KnightMobilityEG, &res[curIndex], &curIndex, KNIGHT_MOBILITY_N);

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

void incParam(int *arr, int n, int value) {
    (*(arr + n)) += value;
    /*for (int i = 0; i < positionsCount; i++) {
        linearEvals[i] += linearEvalPositions[i][n] * value;
    }*/
    for (int i = 0; i < paramsInfluence[n].positionsCount; i++) {
        linearEvals[paramsInfluence[n].positionsList[i]] += linearEvalPositions[paramsInfluence[n].positionsList[i]][n] * value;
    }

    printf("%d\n", paramsInfluence[n].positionsCount);
}

void printParams(char* filename, char* linearFileName) {
    int *params = evalParams;

    int curIndex = 0;

    FILE *f;

    // Linear
    if ((f = fopen(linearFileName, "w")) == NULL) {
        printf("Не удалось открыть файл");
        return;
    }

    for (int i = 0; i < PARAMS_COUNT; ++i) {
        fprintf(f, "%d ", params[i]);
    }

    fclose(f);

    // Standard
    if ((f = fopen(filename, "w")) == NULL) {
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
    printArray("QueenMobilityMG", &params[curIndex], &curIndex, QUEEN_MOBILITY_N, f);
    printArray("RookMobilityMG", &params[curIndex], &curIndex, ROOK_MOBILITY_N, f);
    printArray("BishopMobilityMG", &params[curIndex], &curIndex, BISHOP_MOBILITY_N, f);
    printArray("KnightMobilityMG", &params[curIndex], &curIndex, KNIGHT_MOBILITY_N, f);
    printArray("QueenMobilityEG", &params[curIndex], &curIndex, QUEEN_MOBILITY_N, f);
    printArray("RookMobilityEG", &params[curIndex], &curIndex, ROOK_MOBILITY_N, f);
    printArray("BishopMobilityEG", &params[curIndex], &curIndex, BISHOP_MOBILITY_N, f);
    printArray("KnightMobilityEG", &params[curIndex], &curIndex, KNIGHT_MOBILITY_N, f);

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
    int *prevValues = getValues();

    linearEvalPositions[positionNumber] = malloc(sizeof(double) * PARAMS_COUNT);

    double *linearEval = malloc(PARAMS_COUNT * sizeof(double));
    int *values = malloc(PARAMS_COUNT * sizeof(int));
    memset(values, 0, PARAMS_COUNT * sizeof(int));

    int stage = stageGame(board);
    setValues(values, stage);

    double up = 1000;

    int ev1 = fullEval(board);
    if (ev1 > 0) {
        printf("WRONG\n");
        exit(0);
    }

    int evalParamsCount = 0;
    for (int i = 0; i < PARAMS_COUNT; i++) {
        values[i] = up;
        setValues(values, stage);

        int ev = fullEval(board);

        linearEval[i] = ev / up;
        values[i] = 0;

        linearEvalPositions[positionNumber][i] = linearEval[i];
        evalParamsCount++;
    }

    free(values);
    setValues(prevValues, stage);
    free(prevValues);
}

double getLinearEval(int positionNumber) {
    double eval = 0;
    int *evalSettings = getValues();

    for (int i = 0; i < PARAMS_COUNT; i++) {
        eval += linearEvalPositions[positionNumber][i] * evalSettings[i];
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