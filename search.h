#ifndef SEARCH_H
#define SEARCH_H

#include "types.h"
#include "board.h"
#include "movegen.h"

U16 moves[MAX_PLY][256];

U64 perftTest(Board* board, int depth, int height);

#endif