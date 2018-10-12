#include <stdio.h>
#include <stdlib.h>
#include "uci.h"
#include "board.h"
#include "magic.h"

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
    magicArraysInit();
}