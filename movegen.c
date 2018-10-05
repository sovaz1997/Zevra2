#include "movegen.h"

void movegen(Board* board, uint16_t* moveList) {
    int color = board->color;
    U64 mask = board->pieces[ROOK] & board->colours[color];

    while(mask) {
        int sq = ctz(mask);

        printf("%d\n", sq);

        mask &= ~(1ull << sq);
    }
}