#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <malloc.h>
#include <stdlib.h>

#include "tuning.h"
#include "search.h"

const double K = 150;

void makeTuning() {
    readFens();
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

void readFens(Board* board) {
    char buf[4096];
    char *estr;

    FILE* f = fopen("positions.txt","r");

    SearchInfo searchInfo;
    TimeManager tm = createFixDepthTm(MAX_PLY - 1);
    resetSearchInfo(&searchInfo, tm);

    int posCount = 0;
    const double fadingFactor = 40;

    double errorSums = 0;
    while(1) {
        resetSearchInfo(&searchInfo, tm);

        estr = fgets(buf, sizeof(buf), f);

        if (!estr) {
            break;
        }

        char** res = str_split(estr, ',');

        char* fen = *res;
        int movesToEnd = atoi(*(res + 1));
        double result = atof(*(res + 2));

        double fading = exp(-movesToEnd / fadingFactor);

        setFen(board, fen);

        int eval = quiesceSearch(board, &searchInfo, -MATE_SCORE, MATE_SCORE, 0);

        if (board->color == BLACK) {
            eval = -eval;
        }

        double error = pow(r(eval) - result, 2) * fading;
        errorSums += error;

        if (posCount % 1000 == 0) {
            printf("Positions: %d; Eval: %d; Result: %f; R: %f Fading: %f error: %f\n", posCount, eval, result, r(eval), fading, error);
        }

        free(res);
        ++posCount;
    }

    errorSums /= posCount;

    printf("Error: %f\n", errorSums);

    fclose(f);
}

int setValues(int* values) {
    return 7;
}

int* getValues() {
    int* res = malloc(sizeof(int) * 5);

    res[0] = PAWN_EV;
    res[1] = KNIGHT_EV;
    res[2] = BISHOP_EV;
    res[3] = ROOK_EV;
    res[4] = QUEEN_EV;

    return res;
}