#ifndef MAGIC_H
#define MAGIC_H

#include <stdlib.h>
#include "types.h"
#include "bitboards.h"
#include "board.h"

U64 rookMagicMask[64];
U64 bishopMagicMask[64];

U64 rookPossibleMoves[64][4096];
U64 bishopPossibleMoves[64][1024];

U64 rookPossibleMovesSize[64];
U64 bishopPossibleMovesSize[64];

extern const U64 rookMagic[];
extern const U64 bishopMagic[];


void magicGen();
U64 getAsIndex(U64 bitboard, int index);
int getMagicIndex(U64 configuration, U64 magic, int size);
U64 magicFind(U64 bitboard);
int perfectHashTest(U64 bitboard, U64 magic);
U64 magicRand();

void magicArraysInit();
void preInit();

#endif