#include <stdio.h>
#include <stdlib.h>
#include "bitboards.h"
#include "board.h"
#include "movegen.h"

int main() {
    
    initBitboards();    

    Board* board = (Board*) malloc(sizeof(Board));

    char startpos[] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    
    setFen(board, startpos);
    printBoard(board);

    uint16_t moveList[256];
    
    for(int i = 0; i < 100000000; ++i) {
        movegen(board, moveList);
    }

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