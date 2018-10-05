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

#define bitboardCell(cell_nb) (1ull << (cell_nb))

void initBitboards();
void attacksGen();

void printBitboard(U64 bitboard);
unsigned int square(unsigned int r, unsigned int f);
unsigned int popcount(U64 bitboard);
unsigned int clz(U64 bitboard);
unsigned int ctz(U64 bitboard);

void setBit(uint64_t* bitboard, int sq);
int getBit(uint64_t bitboard, int sq);
void clearBit(uint64_t* bitboard, int sq);
int rankOf(int sq);
int fileOf(int sq);


#endif