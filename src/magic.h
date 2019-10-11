#ifndef MAGIC_H
#define MAGIC_H

#include <stdlib.h>
#include <inttypes.h>
#include "types.h"
#include "bitboards.h"
#include "board.h"

U64 rookMagicMask[64];
U64 bishopMagicMask[64];

U64 rookPossibleMoves[64][4096];
U64 bishopPossibleMoves[64][512];

U64 rookPossibleMovesSize[64];
U64 bishopPossibleMovesSize[64];

extern const U64 rookMagic[];
extern const U64 bishopMagic[];

U64 getAsIndex(U64 bitboard, int index);
int getMagicIndex(U64 configuration, U64 magic, int size);
U64 blockerCut(int from, U64 occu, U64* directionArray, int direction, U64 possibleMoves);
void magicArraysInit();
void preInit();

#endif