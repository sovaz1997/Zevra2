#ifndef BITBOARDS_H
#define BITBOARDS_H

#include <stdio.h>
#include "types.h"

U64 files[8];
U64 ranks[8];

#define bitboardCell(cell_nb) (1ull << (cell_nb))

void initBitboards();
void printBitboard(U64 bitboard);
int square(int r, int f);

#endif