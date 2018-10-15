#ifndef ZOBRIST_H
#define ZOBRIST_H

#include <stdint.h>
#include <stdio.h>
#include "types.h"

extern U64 nextSeed;
U64 zobristKeys[15][64]; //key[piece]][sq]
U64 zobristCastlingKeys[4];
U64 zobristEnpassantKeys[64];
U64 nullMoveKey;

//xorshift*
U64 rand64();
void zobristInit();

#endif