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

//Массивы лучей
U64 plus1[64];
U64 minus1[64];
U64 plus8[64];
U64 minus8[64];
U64 plus7[64];
U64 minus7[64];
U64 plus9[64];
U64 minus9[64];

//Маски
U64 squareBitboard[64];
U64 unSquareBitboard[64];

#define bitboardCell(cell_nb) (1ull << (cell_nb))

//битбоарды для определения возможности рокировки
U64 shortCastlingBitboard[2];
U64 longCastlingBitboard[2];

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
void clearBit(U64* bitboard, int sq);
int rankOf(int sq);
int fileOf(int sq);

#endif