#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "bitboards.h"
#include "board.h"
#include "movegen.h"
#include "magic.h"

void initEngine();

int main() {
    srand(time(0));
    
    
    initEngine();

    Board* board = (Board*) malloc(sizeof(Board));

    //char startpos[] = "rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1";
    //char startpos[] = "4r3/1p4rk/p1pPb2b/2P1p2n/PQ2P2P/5Rp1/1P4K1/5R2 w - - 2 47";
    char startpos[] = "2k5/8/8/8/8/3K4/8/1B6 w - - 0 1";
    
    setFen(board, startpos);
    printBoard(board);

    uint16_t moveList[256];
    
    //for(int i = 0; i < 100000000; ++i) {
        movegen(board, moveList);
    //}

    uint16_t* curMove = moveList;

    while(*curMove) {
        char move[6];
        moveToString(*curMove, move);
        printf("%s\n", move);
        ++curMove;
    }
    

    free(board);

    return 0;
}

void initEngine() {
    initBitboards();
    magicArraysInit();
}