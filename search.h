#ifndef SEARCH_H
#define SEARCH_H

#include <math.h>
#include <time.h>
#include <pthread.h>
#include "types.h"
#include "board.h"
#include "movegen.h"
#include "eval.h"
#include "uci.h"
#include "timemanager.h"
#include "transposition.h"

int SEARCH_COMPLETE;

#define MAX_THREADS_NUM 64

struct SearchArgs {
    Board* board;
    TimeManager tm;
    int threads;
    int threadNumber;
};

int history[2][64][64];

struct SearchInfo {
    U64 nodesCount;
    U16 bestMove;
    Timer timer;
    TimeManager tm;
    U16 killer[2][MAX_PLY + 1];
    U16 secondKiller[2][MAX_PLY + 1];
    int nullMoveSearch;
    int searchTime;
    int* abortSignal;
    int threadNumber;
};

//Тип оценки
enum {
    empty = 0,
    lowerbound = 1,
    upperbound = 2,
    exact = 3
};

extern int FutilityMargin[7];

U16 moves[MAX_THREADS_NUM][MAX_PLY][256];
int movePrice[256];
int mvvLvaScores[7][7];
int lmr[MAX_PLY][64];

int ABORT[MAX_THREADS_NUM];
SearchArgs newArgs[MAX_THREADS_NUM];
Board boards[MAX_THREADS_NUM];
pthread_t searchThreads[MAX_THREADS_NUM];

void* goSMP(void* thread_data);
void* goOtherThread(void* thread_data);
void iterativeDeeping(Board* board, TimeManager tm, int threadNumber);
int search(Board* board, SearchInfo* searchInfo, int alpha, int beta, int depth, int height);
int aspirationWindow(Board* board, SearchInfo* searchInfo, int depth, int score);
int quiesceSearch(Board* board, SearchInfo* searchInfo, int alpha, int beta, int height);
U64 perftTest(Board* board, int depth, int height);
void perft(Board* board, int depth);
void moveOrdering(Board* board, U16* moves, SearchInfo* searchInfo, int height);
void sort(U16* moves, int count);
void initSearch();
void resetSearchInfo(SearchInfo* info, TimeManager tm, int* abortSignal, int threadNumber);
void replaceTransposition(Transposition* tr, Transposition new_tr, int height);
void setAbort(int* signal, int val);
void stopAllThreads();
void clearHistory();
void compressHistory();

#endif