#include "transposition.h"

void setTransposition(Transposition* entry, U64 key, int eval, int evalType, int depth, U16 move, int age) {
    entry->key = key;
    entry->eval = eval;
    entry->evalType = evalType;
    entry->move = move;
    entry->depth = depth;
    entry->age = age;
}

void initTT() {
    ttSize = (1 << 22);
    tt = (Transposition*) malloc(sizeof(Transposition) * ttSize);
    ttIndex = 0;
    for(int i = 0; i < 22; ++i) {
        ttIndex |= (1 << i);
    }
    clearTT();
}

void clearTT() {
    memset(tt, 0, sizeof(Transposition) * ttSize);
    ttAge = 0;
}