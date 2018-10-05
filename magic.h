#ifndef MAGIC_H
#define MAGIC_H

#include <stdlib.h>
#include "types.h"
#include "bitboards.h"

U64 rookMagicMask[64];
U64 bishopMagicMask[64];

void magicGen();
U64 getAsIndex(U64 bitboard, int index);
U64 magicFind(U64 bitboard);
int perfectHashTest(U64 bitboard, U64 magic);
U64 magicRand();

#endif