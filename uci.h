#ifndef UCI_H
#define UCI_H

#include "board.h"
#include "search.h"
#include "eval.h"
#include "types.h"

struct Option {
    int defaultHashSize;
    int minHashSize;
    int maxHashSize;
};

extern char startpos[];
pthread_mutex_t mutex;

int main();
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

#endif