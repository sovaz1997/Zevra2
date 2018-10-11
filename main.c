#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "bitboards.h"
#include "board.h"
#include "movegen.h"
#include "magic.h"
#include "search.h"

void initEngine();

void moveGenTest(Board* board);

int main() {
    srand(time(0));
    
    initEngine();

    Board* board = (Board*) malloc(sizeof(Board));


    char startpos[] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    
    setFen(board, startpos);
    printBoard(board);
    //moveGenTest(board);
    printf("All: %llu\n", perftTest(board, 6, 0));
    free(board);

    return 0;
}

void initEngine() {
    initBitboards();
    magicArraysInit();
}

void moveGenTest(Board* board) {
    uint16_t moves[256];
    movegen(board, moves);

    uint16_t* cur = moves;
    while(*cur) {
        char mv_str[6];
        moveToString(*cur, mv_str);
        printf("%s\n", mv_str);
        ++cur;
    }
    printf("Count: %d\n", cur - moves);
}