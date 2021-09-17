#include "transposition.h"
#include "search.h"

//void setTransposition(Transposition* entry, U64 key, int eval, int evalType, int depth, U16 move, int age, int height) {
//    if (entry->key == 0 || entry->key == key) {
//        entry->key = key;
//
//        int index = getBucketWithLessDepth(entry);
//
//        entry->entity[index].eval = evalToTT(eval, height);
//        entry->entity[index].evalType = evalType;
//        entry->entity[index].move = move;
//        entry->entity[index].depth = depth;
//        entry->entity[index].age = age;
//    }
//
//}

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

void replaceTranspositionEntry(Transposition* addr, TranspositionEntity* newEntry) {
    int replacePriorities[BUCKETS_N];

    for (int i = 0; i < BUCKETS_N; ++i) {
        if(addr->entity[i].age + 5 < ttAge || !addr->entity[i].evalType) {
            replacePriorities[i] = MAX_PLY * 2 - addr->entity[i].depth;
            continue;
        }

        if(newEntry->depth >= addr->entity[i].depth) {
            if(newEntry->evalType == upperbound && addr->entity[i].evalType != upperbound && addr->entity[i].evalType != upperbound) {
                replacePriorities[i] = -1;
                continue;
            }

            replacePriorities[i] = MAX_PLY * 2 - addr->entity[i].depth;
        }
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

int getMaxDepthBucket(Transposition* entry) {
    uint32_t depth = 0;
    int result = -1;

    for (int i = 0; i < BUCKETS_N; ++i) {
        if (entry->entity[i].depth > depth) {
            depth = entry->entity[i].depth;
            result = i;
        }
    }

    return result;
}