#include <stdio.h>
#include <stdlib.h>
#include "bitboards.h"
#include "board.h"

int main() {
    
    initBitboards();    

    Board* board = (Board*) malloc(sizeof(Board));

    char startpos[] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    
    setFen(board, startpos);
    printBoard(board);

    free(board);

    return 0;
}