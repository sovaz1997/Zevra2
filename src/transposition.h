#ifndef TRANSPOSITION_H
#define TRANSPOSITION_H

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "types.h"
#include "bitboards.h"

enum {
    BUCKETS_N = 2,
};

struct TranspositionEntity {
    S16 eval;
    U8 depth;
    // S8 age;
    U8 evalType;
    U16 move;
    U64 key;
};

struct Transposition {
//    S16 eval[BUCKETS_N];
//    U8 depth[BUCKETS_N];
//    S8 age[BUCKETS_N];
//    U8 evalType[BUCKETS_N];
//    U16 move[BUCKETS_N];
    TranspositionEntity entity[BUCKETS_N];
};

Transposition* tt;
U64 ttSize;
double ttFilledSize;
U64 ttIndex;
int ttAge;

void initTT(int size);
void reallocTT(int size);
void clearTT();
void replaceTranspositionEntry(Transposition* addr, TranspositionEntity* newEntry, U64 key);
U64 sizeToTTCount(U64 size);
int evalToTT(int eval, int height);
int evalFromTT(int eval, int height);
int getMaxDepthBucket(Transposition* entry, U64 key);

#endif