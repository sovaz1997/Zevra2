#ifndef TRANSPOSITION_H
#define TRANSPOSITION_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "types.h"
#include "bitboards.h"

Transposition* tt;
U64 ttSize;
U64 ttFilledSize;
U64 ttIndex;
int ttAge;

struct Transposition {
    U64 key;
    S16 eval;
    U8 depth;
    S8 age;
    U8 evalType;
    U16 move;
};

void setTransposition(Transposition* entry, U64 key, int eval, int evalType, int depth, U16 move, int age);
void initTT(int size);
void reallocTT(int size);
void clearTT();
void replaceTranspositionEntry(Transposition* addr, Transposition* newEntry);
U64 sizeToTTCount(U64 size);

#endif