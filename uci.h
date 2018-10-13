#ifndef UCI_H
#define UCI_H

#include "board.h"
#include "search.h"
#include "eval.h"

extern char startpos[];

void uciInterface(Board* board);
void printEngineInfo();
void printScore(int score);

#endif