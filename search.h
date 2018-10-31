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

int ABORT;
int SEARCH_COMPLETE;

struct SearchArgs {
    Board* board;
    TimeManager tm;
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
};

//Тип оценки
enum {
    empty = 0,
    lowerbound = 1,
    upperbound = 2,
    exact = 3
};

extern int FutilityMargin[7];

U16 moves[MAX_PLY][256];
int movePrice[MAX_PLY][256];
int mvvLvaScores[7][7];
int lmr[MAX_PLY][64];

//Эвристики
static int FutilityPruningAllow = 1;
static int NullMovePruningAllow = 1;
static int LmrPruningAllow = 1;
static int HistoryPruningAllow = 1;

void* go(void* thread_data);
void iterativeDeeping(Board* board, TimeManager tm);
int search(Board* board, SearchInfo* searchInfo, int alpha, int beta, int depth, int height);
int aspirationWindow(Board* board, SearchInfo* searchInfo, int depth, int score);
int quiesceSearch(Board* board, SearchInfo* searchInfo, int alpha, int beta, int height);
U64 perftTest(Board* board, int depth, int height);
void perft(Board* board, int depth);
void moveOrdering(Board* board, U16* moves, SearchInfo* searchInfo, int height);
void sort(U16* moves, int count, int height);
void initSearch();
void resetSearchInfo(SearchInfo* info, TimeManager tm);
void replaceTransposition(Transposition* tr, Transposition new_tr, int height);
void setAbort(int val);
void clearHistory();
void compressHistory();

#endif