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


    char startpos[] = "r3kbnr/8/8/8/8/3q3b/7B/RN1QKBNR b KQkq - 2 3";
    //char startpos[] = "r1b1k2r/p3bp1p/2nppqpn/1pp5/2PP4/PPNBPN1P/5PP1/R1BQK2R w KQkq - 1 10";
    
    setFen(board, startpos);
    printBoard(board);
    //moveGenTest(board);
    printf("All: %llu\n", perftTest(board, 5, 0));
    printBoard(board);
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