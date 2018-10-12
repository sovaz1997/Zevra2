#ifndef UCI_H
#define UCI_H

#include "board.h"
#include "search.h"

extern char startpos[];

void uciInterface(Board* board);
void printEngineInfo();

#endif