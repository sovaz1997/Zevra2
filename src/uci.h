#ifndef UCI_H
#define UCI_H

#include "board.h"
#include "search.h"
#include "eval.h"
#include "types.h"

extern int SHOULD_GENERATE_DATASET;
extern int NNUE_ENABLED;
extern int SHOULD_HIDE_SEARCH_INFO_LOGS;
extern int temperature;

struct Option {
    int defaultHashSize;
    int minHashSize;
    int maxHashSize;
    int defaultTemperature;
    int minTemperature;
    int maxTemperature;
};

extern Option option;

extern char startpos[];

// Compile-time switch: expose the SPSA-tunable search/TM parameters as UCI
// options. Off in release builds so the tuning knobs stay hidden; enable with
// `make tune` (or -DTUNING_ENABLED=1) for local SPSA runs.
#ifndef TUNING_ENABLED
#define TUNING_ENABLED 0
#endif
extern pthread_mutex_t mutex;

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