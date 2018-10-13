#ifndef SEARCH_H
#define SEARCH_H

#include <time.h>
#include "types.h"
#include "board.h"
#include "movegen.h"
#include "eval.h"
#include "uci.h"

struct SearchInfo {
    U64 nodesCount;
    U16 bestMove;
};

U16 moves[MAX_PLY][256];
int movePrice[256];
int mvvLvaScores[7][7];

void iterativeDeeping(Board* board, int depth);
int search(Board* board, SearchInfo* searchInfo, int alpha, int beta, int depth, int height);
int quiesceSearch(Board* board, SearchInfo* searchInfo, int alpha, int beta, int height);
U64 perftTest(Board* board, int depth, int height);
void perft(Board* board, int depth);
void moveOrdering(Board* board, U16* moves, SearchInfo* searchInfo);
void sort(U16* moves, int count);
void initSearch();

#endif