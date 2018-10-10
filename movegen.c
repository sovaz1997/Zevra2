#include "movegen.h"

void movegen(Board* board, uint16_t* moveList) {
    int color = board->color;


    U64 our = board->colours[color];
    U64 enemy = board->colours[!color];

    U64 mask = board->pieces[ROOK] & our;
    U64 occu = (our | enemy);

    while(mask) {
        int from = ctz(mask);
        U64 possibleMoves = rookPossibleMoves[from][getMagicIndex(occu & rookMagicMask[from] & ~bitboardCell(from), rookMagic[from], rookPossibleMovesSize[from])];
        moveList = genMovesFromBitboard(from, possibleMoves & ~our, moveList);
        clearBit(&mask, from);
    }

    mask = board->pieces[BISHOP] & our;

    while(mask) {
        int from = ctz(mask);
        U64 possibleMoves = bishopPossibleMoves[from][getMagicIndex(occu & bishopMagicMask[from] & ~bitboardCell(from), bishopMagic[from], bishopPossibleMovesSize[from])];
        moveList = genMovesFromBitboard(from, possibleMoves & ~our, moveList);
        clearBit(&mask, from);
    }

    mask = board->pieces[QUEEN] & our;

    while(mask) {
        int from = ctz(mask);
        U64 possibleMoves = rookPossibleMoves[from][getMagicIndex(occu & rookMagicMask[from] & ~bitboardCell(from), rookMagic[from], rookPossibleMovesSize[from])];
        possibleMoves |= bishopPossibleMoves[from][getMagicIndex(occu & bishopMagicMask[from] & ~bitboardCell(from), bishopMagic[from], bishopPossibleMovesSize[from])];
        moveList = genMovesFromBitboard(from, possibleMoves & ~our, moveList);
        clearBit(&mask, from);
    }

    mask = board->pieces[KNIGHT] & our;

    while(mask) {
        int from = ctz(mask);

        U64 possibleMoves = knightAttacks[from] & ~board->colours[color];
        moveList = genMovesFromBitboard(from, possibleMoves & ~our, moveList);
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