#include "mcts.h"
#include "search.h"

U16 movesCash1[512];
U16 movesCash2[512];

int MCTSSearch(Board* board) {
    simulate(board, movesCash1);
}

double simulate(Board* board, U16* movesCash) {
    if (isDraw(board)) {
        return 0.5;
    }

    int movesCount = generatePossibleMoves(board, movesCash);

    if (movesCount == 0) {
        return inCheck(board, board->color) ? 0 : 0.5;
    }

    int randMove = rand() % movesCount;

    Undo undo;
    U16 move = movesCash[randMove];
    printMove(move);
    makeMove(board, move, &undo);
    double res = 1 - simulate(board, movesCash);
    unmakeMove(board, move, &undo);
    return res;
}

int generatePossibleMoves(Board* board, U16* moves) {
    movegen(board, moves);

    U16* curMove = moves;
    U16* possibleMove = curMove;
    Undo undo;

    int movesCount = 0;
    while(*curMove) {
        makeMove(board, *curMove, &undo);

        if(!inCheck(board, !board->color)) {
            possibleMove[movesCount] = *curMove;
            ++movesCount;
        }

        (*curMove)++;
        unmakeMove(board, *curMove, &undo);
    }

    possibleMove[movesCount] = 0;
    return movesCount;
}