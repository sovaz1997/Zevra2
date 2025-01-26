#include "transposition.h"
#include "search.h"

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

void replaceTranspositionEntry(Transposition* addr, TranspositionEntity* newEntry, U64 key) {
    int replacePriorities[1];

    if(addr->entity[0].age + 5 < ttAge || !addr->entity[0].evalType) {
        replacePriorities[0] = MAX_PLY * 2 - addr->entity[0].depth;
    } else {
        if(newEntry->depth >= addr->entity[0].depth) {
            if(newEntry->evalType == upperbound && addr->entity[0].evalType != upperbound) {
                replacePriorities[0] = -1;
            } else {
                replacePriorities[0] = MAX_PLY * 2 - addr->entity[0].depth;
            }
        }
    }

    int index = -1;
    int maxPriority = -1;

    if (replacePriorities[0] > maxPriority) {
        index = 0;
        maxPriority = replacePriorities[0];
    }

    if (index != -1) {
        if (!addr->entity[index].evalType) {
            ttFilledSize++;
        }

        addr->entity[index] = *newEntry;
    }
}

U64 sizeToTTCount(U64 size) {
    U64 count;
    size *= (1024 * 1024);
    for(count = 1; count * sizeof(Transposition) <= size; count *= 2);

    if(count * sizeof(Transposition) > size)
        count /= 2;

    return count;
}

int evalToTT(int eval, int height) {
    if(eval > MATE_SCORE - 100)
        return eval + height;
    else if(eval < -MATE_SCORE + 100)
        return eval - height;

    return eval;
}

int evalFromTT(int eval, int height) {
    if(eval > MATE_SCORE - 100)
        return eval - height;
    else if(eval < -MATE_SCORE + 100)
        return eval + height;
        
    return eval;
}

int getMaxDepthBucket(Transposition* entry, U64 key) {
    uint32_t depth = 0;
    int result = -1;

    if (entry->entity[0].depth > depth && key == entry->entity[0].key) {
        depth = entry->entity[0].depth;
        result = 0;
    }

    return result;
}