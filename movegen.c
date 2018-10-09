#include "movegen.h"

void movegen(Board* board, uint16_t* moveList) {
    int color = board->color;

    U64 mask = board->pieces[ROOK] & board->colours[color];
    U64 occu = (board->colours[WHITE] | board->colours[BLACK]);

    while(mask) {
        int from = ctz(mask);
        printf("%lld\n", occu & rookMagicMask[from] & ~bitboardCell(from));
        //printBitboard(occu & rookMagicMask[from] & ~bitboardCell(from));
        U64 possibleMoves = rookPossibleMoves[from][getMagicIndex(occu & rookMagicMask[from] & ~bitboardCell(from), rookMagic[from], rookPossibleMovesSize[from])];
        
        printBitboard(possibleMoves);
        moveList = genMovesFromBitboard(from, possibleMoves, moveList);
        clearBit(&mask, from);
    }

    mask = board->pieces[KNIGHT] & board->colours[color];

    while(mask) {
        int from = ctz(mask);

        U64 possibleMoves = knightAttacks[from] & ~board->colours[color];
        moveList = genMovesFromBitboard(from, possibleMoves, moveList);
        clearBit(&mask, from);
    }

    *moveList = 0;
}

uint16_t* genMovesFromBitboard(int from, U64 bitboard, uint16_t* moveList) {
    while(bitboard) {
        int to = ctz(bitboard);
        *(moveList++) = MakeMove(from, to, NORMAL_MOVE);
        clearBit(&bitboard, to);
    }
    return moveList;
}