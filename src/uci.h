#ifndef UCI_H
#define UCI_H

#include "board.h"
#include "search.h"
#include "eval.h"
#include "types.h"

extern int SHOULD_GENERATE_DATASET;
extern int NNUE_ENABLED;
extern int SHOULD_HIDE_SEARCH_INFO_LOGS;
extern int shouldUseNNUE;
int temperature;

struct Option {
    int defaultHashSize;
    int minHashSize;
    int maxHashSize;
    int defaultTemperature;
    int minTemperature;
    int maxTemperature;
    int shouldUseNNUE;
    int defaultShouldUseNNUE;
};

Option option;

extern char startpos[];
extern const int TUNING_ENABLED;
pthread_mutex_t mutex;

int main(int argc, char** argv);
void makeCommand();
void printPV(Board* board, int depth, U16 bestMove);
void printEngineInfo();
void printScore(int score);
void printSearchInfo(SearchInfo* info, Board* board, int depth, int eval, int evalType);
void input(char* str);
int findMove(char* move, Board* board);
void readyok();
void initOption();
void initEngine();

int strEquals(char* str1, char* str2);
int strStartsWith(char* str, char* key);

#endif