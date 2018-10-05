#include "movegen.h"

void movegen(Board* board, uint16_t* moveList) {
    int color = board->color;

    /*U64 mask = board->pieces[ROOK] & board->colours[color];

    while(mask) {
        int sq = ctz(mask);

        printf("%d\n", sq);

        mask &= ~(1ull << sq);
    }*/

    U64 mask = board->pieces[KNIGHT] & board->colours[color];

    while(mask) {
        int from = ctz(mask);

        U64 possibleMoves = knightAttacks[from] & ~board->colours[color];
        
        while(possibleMoves) {
            int to = ctz(possibleMoves);
            *(moveList++) = MakeMove(from, to, NORMAL_MOVE);
            clearBit(&possibleMoves, to);
        }

        clearBit(&mask, from);
    }

    *moveList = 0;
}