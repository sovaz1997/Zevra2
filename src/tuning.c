#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <malloc.h>
#include <stdlib.h>

#include "tuning.h"
#include "search.h"

const double K = 150;
const int PARAMS_COUNT = 8 + 7 * 64 + 28 + 15 + 14 + 8 + 8 + 4;

struct TuningPosition {
    char fen[512];
    int movesToEnd;
    double result;

};

void makeTuning(Board* board) {
    loadPositions();

    int* curValues = getValues();

    const int changeFactor = 1;

    double E = fun(board);

    while(1) {
        int improved = 0;
        int iterations = 0;
        for (int i = 0; i < PARAMS_COUNT; i++) {

            int tmpParam = curValues[i];
            changeParam(i, curValues[i] + changeFactor);

            double newE = fun(board);

            if (newE < E) {
                improved = 1;
                curValues[i] += changeFactor;
                E = newE;
                printParams();
                iterations++;
                printf("NewE: %.7f; index: %d; value: %d\n", E, i, curValues[i]);
            } else {
                changeParam(i, curValues[i] - changeFactor);

                newE = fun(board);

                if (newE < E) {
                    curValues[i] -= changeFactor;
                    improved = 1;
                    E = newE;
                    printParams();
                    iterations++;
                    printf("NewE: %.7f; index: %d; value: %d\n", E, i, curValues[i]);
                } else {
                    changeParam(i, tmpParam);
                }
            }
        }

        printf("Iterations: %d/%d\n", iterations, PARAMS_COUNT);

        if (!improved) {
            break;
        }
    }

    for (int i = 0; i < PARAMS_COUNT; i++) {
        printf("%d ", curValues[i]);
    }
}

int positionsCount = 0;
TuningPosition* positions;

void loadPositions() {
    FILE* f = fopen("positions.txt","r");

    char buf[4096];
    char *estr;

    while(1) {
        estr = fgets(buf, sizeof(buf), f);

        if (!estr) {
            break;
        }

        ++positionsCount;

        char** res = str_split(estr, ',');

        char* fen = *res;
        int movesToEnd = atoi(*(res + 1));
        double result = atof(*(res + 2));

        if (positionsCount == 1) {
            positions = malloc(sizeof(TuningPosition) * 120000000);
        } else {
            // positions = realloc(positions, sizeof(TuningPosition) * positionsCount);
        }

        strcpy(positions[positionsCount - 1].fen, fen);
        if (positionsCount % 1000 == 0) {
            printf("Pos count: %d\n", positionsCount);
        }

        positions[positionsCount - 1].result = result;
        positions[positionsCount - 1].movesToEnd = movesToEnd;

        for(int i = 0; i < 3; i++) {
            free(res[i]);
        }

        free(res);
    }
    printf("Pos count: %d\n", positionsCount);

    fclose(f);

}

char** str_split(char* a_str, const char a_delim)
{
    char** result    = 0;
    size_t count     = 0;
    char* tmp        = a_str;
    char* last_comma = 0;
    char delim[2];
    delim[0] = a_delim;
    delim[1] = 0;

    /* Count how many elements will be extracted. */
    while (*tmp)
    {
        if (a_delim == *tmp)
        {
            count++;
            last_comma = tmp;
        }
        tmp++;
    }

    /* Add space for trailing token. */
    count += last_comma < (a_str + strlen(a_str) - 1);

    /* Add space for terminating null string so caller
       knows where the list of returned strings ends. */
    count++;

    result = malloc(sizeof(char*) * count);

    if (result)
    {
        size_t idx  = 0;
        char* token = strtok(a_str, delim);

        while (token)
        {
            assert(idx < count);
            *(result + idx++) = strdup(token);
            token = strtok(0, delim);
        }
        assert(idx == count - 1);
        *(result + idx) = 0;
    }

    return result;
}

double r(int eval) {
    return 1. / (1. + exp(-eval / K));
}

double fun(Board* board) {
    SearchInfo searchInfo;
    TimeManager tm = createFixDepthTm(MAX_PLY - 1);
    resetSearchInfo(&searchInfo, tm);

    int posCount = 0;
    const double fadingFactor = 40;

    double errorSums = 0;



    for(int i = 0; i < positionsCount; i++) {
        resetSearchInfo(&searchInfo, tm);

        char* fen = positions[i].fen;
        int movesToEnd = positions[i].movesToEnd;
        double result = positions[i].result;

        double fading = exp(-movesToEnd / fadingFactor);

        setFen(board, fen);

        int eval = quiesceSearch(board, &searchInfo, -MATE_SCORE, MATE_SCORE, 0);

        if (board->color == BLACK) {
            eval = -eval;
        }

        double error = pow(r(eval) - result, 2) * fading;
        errorSums += error;

        ++posCount;
    }

    errorSums /= posCount;

    return errorSums;
}

void setValues(int* values) {
    PAWN_EV = values[0];
    KNIGHT_EV = values[1];
    BISHOP_EV = values[2];
    ROOK_EV = values[3];
    QUEEN_EV = values[4];

    KingDangerFactor = values[5];
    RookOnOpenFileBonus = values[6];
    RookOnPartOpenFileBonus = values[7];


    int curIndex = 8;

    // PST
    transferPST(&values[curIndex], pawnPST, &curIndex);
    transferPST(&values[curIndex], knightPST, &curIndex);
    transferPST(&values[curIndex], bishopPST, &curIndex);
    transferPST(&values[curIndex], rookPST, &curIndex);
    transferPST(&values[curIndex], queenPST, &curIndex);
    transferPST(&values[curIndex], kingPST, &curIndex);
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


    // re-init due to dependent eval
    initDependencyEval();
}

int* transfer(int* from, int* to, int* curIndex, int length) {
    for (int i = 0; i < length; i++) {
        to[i] = *(from + i);
        (*curIndex)++;
    }
}

int* transferPST(int* from, int* to, int* curIndex) {
    transfer(from, to, curIndex, 64);
}

int* getValues() {
    int* res = (int*)malloc(sizeof(int) * PARAMS_COUNT);

    res[0] = PAWN_EV;
    res[1] = KNIGHT_EV;
    res[2] = BISHOP_EV;
    res[3] = ROOK_EV;
    res[4] = QUEEN_EV;

    res[5] = KingDangerFactor;
    res[6] = RookOnOpenFileBonus;
    res[7] = RookOnPartOpenFileBonus;

    int curIndex = 8;

    // PST
    transferPST(pawnPST, &res[curIndex], &curIndex);
    transferPST(knightPST, &res[curIndex], &curIndex);
    transferPST(bishopPST, &res[curIndex], &curIndex);
    transferPST(rookPST, &res[curIndex], &curIndex);
    transferPST(queenPST, &res[curIndex], &curIndex);
    transferPST(kingPST, &res[curIndex], &curIndex);
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

    return res;
}

void changeParam(int n, int value) {
    int* params = getValues();
    *(params + n) = value;
    setValues(params);
    free(params);
}

void printParams() {
    int* params = getValues();

    int curIndex = 0;

    FILE* f;
    char name[] = "weights.txt";

    if ((f = fopen(name, "w")) == NULL) {
        printf("Не удалось открыть файл");
        return;
    }

    printArray("PAWN_EV", &params[curIndex], &curIndex, 1, f);
    printArray("KNIGHT_EV", &params[curIndex], &curIndex, 1, f);
    printArray("BISHOP_EV", &params[curIndex], &curIndex, 1, f);
    printArray("ROOK_EV", &params[curIndex], &curIndex, 1, f);
    printArray("QUEEN_EV", &params[curIndex], &curIndex, 1, f);
    printArray("KingDangerFactor", &params[curIndex], &curIndex, 1, f);
    printArray("RookOnOpenFileBonus", &params[curIndex], &curIndex, 1, f);
    printArray("RookOnPartOpenFileBonus", &params[curIndex], &curIndex, 1, f);

    printPST("pawnPST", &params[curIndex], &curIndex, f);
    printPST("knightPST", &params[curIndex], &curIndex, f);
    printPST("bishopPST", &params[curIndex], &curIndex, f);
    printPST("rookPST", &params[curIndex], &curIndex, f);
    printPST("queenPST", &params[curIndex], &curIndex, f);
    printPST("kingPST", &params[curIndex], &curIndex, f);
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

    fclose(f);
}

void printPST(char* name, int* pst, int* curIndex, FILE* f) {
    // printf("%s\n", name);
    fprintf(f, "%s\n", name);
    for (int i = 0; i < 64; ++i, (*curIndex)++) {
       //  printf("%d, ", pst[i]);
        fprintf(f, "%d, ", pst[i]);
        if (i > 0 && (i + 1) % 8 == 0) {
          //   printf("\n");
            fprintf(f, "\n");
        }
    }
}

void printArray(char* name, int* arr, int* curIndex, int length, FILE* f) {
    if (length == 1) {
        // printf("%s: %d\n", name, arr[0]);
        fprintf(f, "%s: %d\n", name, arr[0]);
        (*curIndex)++;
        return;
    }

    // printf("%s\n", name);
    fprintf(f, "%s\n", name);
    for (int i = 0; i < length; ++i, (*curIndex)++) {
        // printf("%d, ", arr[i]);
        fprintf(f, "%d, ", arr[i]);
    }
    // printf("\n");
    fprintf(f, "\n");
}