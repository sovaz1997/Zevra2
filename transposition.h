#ifndef TRANSPOSITION_H
#define TRANSPOSITION_H

#include <stdlib.h>
#include <string.h>
#include "types.h"
#include "bitboards.h"

Transposition* tt;
int ttSize;
U64 ttIndex;

struct Transposition {
    U64 key;
    int eval;
    int depth;
    U8 evalType;
    U16 move;
};

void setTransposition(Transposition* entry, U64 key, int eval, int evalType, int depth, U16 move);
void initTT();
void clearTT();

#endif