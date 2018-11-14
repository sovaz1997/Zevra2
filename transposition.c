#include "transposition.h"

void setTransposition(Transposition* entry, U64 key, int eval, int evalType, int depth, U16 move, int age) {
    entry->key = key;
    entry->eval = eval;
    entry->evalType = evalType;
    entry->move = move;
    entry->depth = depth;
    entry->age = age;
}

void initTT(int size) {
    ttSize = sizeToTTCount(size);
    tt = (Transposition*) malloc(sizeof(Transposition) * ttSize);
    ttIndex = ttSize - 1;
    clearTT();
}

void reallocTT(int size) {
    ttSize = sizeToTTCount(size);
    tt = (Transposition*) realloc(tt, sizeof(Transposition) * ttSize);
    ttIndex = ttSize - 1;
    clearTT();
}

void clearTT() {
    memset(tt, 0, sizeof(Transposition) * ttSize);
    ttAge = 0;
    ttFilledSize = 0;
}

void replaceTranspositionEntry(Transposition* addr, Transposition* newEntry) {
    if(!addr->evalType) {
        ++ttFilledSize;
    }
    *addr = *newEntry;
}

U64 sizeToTTCount(U64 size) {
    U64 count;
    size *= (1024 * 1024);
    for(count = 1; count * sizeof(Transposition) <= size; count *= 2);

    if(count * sizeof(Transposition) > size) {
        count /= 2;
    }

    return count;
}