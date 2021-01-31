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

volatile int ABORT;
volatile int SEARCH_COMPLETE;

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
    U16 killer[MAX_PLY + 1][2];
    int nullMoveSearch;
    int searchTime;
    int selDepth;
};

//Eval type
enum {
    empty = 0,
    lowerbound = 1,
    upperbound = 2,
    exact = 3
};

static const int FutilityStep = 50;
static const int ReverseFutilityStep = 90;
static const int RazorMargin = 300;

U16 moves[MAX_PLY][256];
int movePrice[MAX_PLY][256];
int mvvLvaScores[7][7];
int lmr[MAX_PLY][64];

//Heuristics control
static const int FutilityPruningAllow = 1;
static const int NullMovePruningAllow = 1;
static const int LmrPruningAllow = 1;
static const int ReverseFutilityPruningAllow = 1;
static const int RazoringPruningAllow = 1;

void* go(void* thread_data);
SearchInfo iterativeDeeping(Board* board, TimeManager tm);
int search(Board* board, SearchInfo* searchInfo, int alpha, int beta, int depth, int height);
int aspirationWindow(Board* board, SearchInfo* searchInfo, int depth, int score);
int quiesceSearch(Board* board, SearchInfo* searchInfo, int alpha, int beta, int height);
U64 perftTest(Board* board, int depth, int height);
void perft(Board* board, int depth);
void* perftThreads(void* perftArgs);
void moveOrdering(Board* board, U16* mvs, SearchInfo* searchInfo, int height, int depth);
void initSearch();
void resetSearchInfo(SearchInfo* info, TimeManager tm);
void replaceTransposition(Transposition* tr, Transposition new_tr);
void setAbort(int val);
void clearHistory();
void compressHistory();
void movePick(int moveNumber, int height);

#endif