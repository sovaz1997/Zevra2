#ifndef BITBOARDS_H
#define BITBOARDS_H

#include <stdio.h>
#include <string.h>
#include "types.h"

U64 files[8];
U64 ranks[8];

U64 rookAttacks[64];
U64 bishopAttacks[64];
U64 knightAttacks[64];
U64 kingAttacks[64];
U64 pawnMoves[2][64];
U64 pawnAttacks[2][64];

//Beams
U64 plus1[64];
U64 minus1[64];
U64 plus8[64];
U64 minus8[64];
U64 plus7[64];
U64 minus7[64];
U64 plus9[64];
U64 minus9[64];

//Masks
U64 squareBitboard[64];
U64 unSquareBitboard[64];


#define bitboardCell(cell_nb) (1ull << (cell_nb))

//bitboards for castlings
U64 shortCastlingBitboard[2];
U64 longCastlingBitboard[2];

U64 squareTable[8][8];
U64 rankOfTable[64];
U64 fileOfTable[64];

void initBitboards();
void attacksGen();

void printBitboard(U64 bitboard);
unsigned int square(unsigned int r, unsigned int f);
unsigned int popcount(U64 bitboard);
unsigned int clz(U64 bitboard);
unsigned int ctz(U64 bitboard);
unsigned int firstOne(U64 bitboard);
unsigned int lastOne(U64 bitboard);

void setBit(U64* bitboard, int sq);
int getBit(U64 bitboard, int sq);
int getBit8(U8 bitboard, int nb);
void clearBit(U64* bitboard, int sq);
int rankOf(int sq);
int fileOf(int sq);

// Helpers
#define PERSPECTIVE_MASK 0b111000

#endif