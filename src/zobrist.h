#ifndef ZOBRIST_H
#define ZOBRIST_H

#include <stdint.h>
#include <stdio.h>
#include "types.h"

extern U64 nextSeed;
extern U64 zobristKeys[15][64]; //key[piece][sq]
extern U64 zobristCastlingKeys[4];
extern U64 zobristEnpassantKeys[64];
extern U64 nullMoveKey;
extern U64 otherSideKey;

U64 rand64();
void zobristInit();

#endif