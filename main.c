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
void perft(Board* board, int depth);

int main() {
    srand(time(0));
    
    initEngine();

    Board* board = (Board*) malloc(sizeof(Board));


    char startpos[] = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";
    
    setFen(board, startpos);

    perft(board, 9);
    
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

void perft(Board* board, int depth) {
    for(int i = 1; i <= depth; ++i) {
        clock_t start = clock();
        U64 nodes = perftTest(board, i, 0);
        clock_t end = clock();
        if(!(end - start)) {
            end = start + 1;
        }
        
        printf("Perft %d: %llu; speed: %d\n", i, nodes, nodes / (end - start));
    }
}