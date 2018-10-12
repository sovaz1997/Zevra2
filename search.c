#include "search.h"

U64 perftTest(Board* board, int depth, int height) {
    if(!depth) {
        return 1;
    }

    movegen(board, moves[height]);

    U64 result = 0;
    U16* curMove = moves[height];
    Undo undo;
    while(*curMove) {
        makeMove(board, *curMove, &undo);

        U64 count = 0;
        if(!inCheck(board, !board->color)) {
            count = perftTest(board, depth - 1, height + 1);

            if(!height) {
                char mv[6];
                moveToString(*curMove, mv);
                for(int i = 0; i < height; ++i) {
                    printf(" ");
                }
                printf("%s: %llu\n", mv, count);
            }
        }

        result += count;

        unmakeMove(board, *curMove, &undo);

        ++curMove;
    }

    return result;
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