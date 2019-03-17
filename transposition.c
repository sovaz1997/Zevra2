#include "transposition.h"

void setTransposition(Transposition* entry, U64 key, int eval, int evalType, int depth, U16 move, int age, int height) {
    entry->key = key;
    entry->eval = evalToTT(eval, height);
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
    void* tmp = realloc(tt, sizeof(Transposition) * ttSize);
    if(tmp) {
        tt = tmp;
        ttIndex = ttSize - 1;
        clearTT();
    }

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

int evalToTT(int eval, int height) {
    if(eval > MATE_SCORE - 100) {
        return eval + height;
    } else if(eval < -MATE_SCORE + 100) {
        return eval - height;
    }

    return eval;
}

int evalFromTT(int eval, int height) {
    if(eval > MATE_SCORE - 100) {
        return eval - height;
    } else if(eval < -MATE_SCORE + 100) {
        return eval + height;
    }

    return eval;
}