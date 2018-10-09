#include "search.h"

U64 perftTest(Board* board, int depth, int height) {
    if(!depth) {
        return 1;
    }

    movegen(board, moves[height]);

    U64 result = 0;
    int* curMove = moves[height];
    Undo undo;
    while(*curMove) {
        makeMove(board, *curMove, &undo);
        result += perftTest(board, depth - 1, height + 1);
        unmakeMove(board, *curMove, &undo);

        ++curMove;
    }

    return result;
}