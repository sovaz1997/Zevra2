#include <stdio.h>
#include <stdlib.h>
#include "uci.h"
#include "board.h"
#include "magic.h"
#include "zobrist.h"

void initEngine();

int main() {
    initEngine();
    Board* board = (Board*) malloc(sizeof(Board));
    uciInterface(board);    
    free(board);

    return 0;
}

void initEngine() {
    initBitboards();
    zobristInit();
    magicArraysInit();
    initSearch();
}