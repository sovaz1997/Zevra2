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
    int replacePriorities[BUCKETS_N];

    for (int i = 0; i < BUCKETS_N; ++i) {
            if(key == newEntry->key && newEntry->evalType == upperbound && addr->entity[i].evalType != upperbound && addr->entity[i].evalType) {
                replacePriorities[i] = -1;
                continue;
            }

            replacePriorities[i] = MAX_PLY * 2 - addr->entity[i].depth;
    }

    int index = -1;
    int maxPriority = -1;

    for (int i = 0; i < BUCKETS_N; ++i) {
        if (replacePriorities[i] > maxPriority) {
            index = i;
            maxPriority = replacePriorities[i];
        }
    }

    if (index != -1) {
        if (!addr->entity[index].evalType) {
            ttFilledSize += 1. / BUCKETS_N;
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

    for (int i = 0; i < BUCKETS_N; ++i) {
        if (entry->entity[i].depth > depth && key == entry->entity[i].key) {
            depth = entry->entity[i].depth;
            result = i;
        }
    }

    return result;
}