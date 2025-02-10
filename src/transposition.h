#ifndef TRANSPOSITION_H
#define TRANSPOSITION_H

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "types.h"
#include "bitboards.h"

struct Transposition {
    S16 eval;
    U8 depth;
    S8 age;
    U8 evalType;
    U16 move;
    U64 key;
};

Transposition nullTransposition;

Transposition* tt;
U64 ttSize;
double ttFilledSize;
U64 ttIndex;
int ttAge;

void initTT(int size);
void reallocTT(int size);
void clearTT();
void replaceTranspositionEntry(Transposition* newEntry, U64 key);
U64 sizeToTTCount(U64 size);
int evalToTT(int eval, int height);
int evalFromTT(int eval, int height);
Transposition* getTTEntry(U64 key);

#endif