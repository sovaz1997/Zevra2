#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "bitboards.h"
#include "board.h"
#include "movegen.h"
#include "magic.h"
#include "search.h"

void initEngine();

int main() {
    srand(time(0));
    
    initEngine();

    Board* board = (Board*) malloc(sizeof(Board));

    char startpos[] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    
    setFen(board, startpos);
    printBoard(board);
    printf("%llu\n", perftTest(board, 6, 0));
    free(board);

    return 0;
}

void initEngine() {
    initBitboards();
    magicArraysInit();
}