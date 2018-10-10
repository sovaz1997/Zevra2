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
        if(!height) {
            char mv[6];
            moveToString(*curMove, mv);
            printf("%s\n", mv);
        }
        
        makeMove(board, *curMove, &undo);
        result += perftTest(board, depth - 1, height + 1);
        unmakeMove(board, *curMove, &undo);

        ++curMove;
    }

    return result;
}