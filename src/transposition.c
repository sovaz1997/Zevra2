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

void replaceTranspositionEntry(Transposition* newEntry, U64 key) {
  	Transposition* addr = &tt[key & ttIndex];
    int shouldReplace = 0;

    if (addr->age != ttAge || !addr->evalType || addr->evalType == upperbound && newEntry->evalType != upperbound) {
    	shouldReplace = 1;
    } else {
        if(newEntry->depth >= addr->depth) {
            if(newEntry->evalType == upperbound && addr->evalType != upperbound) {
                shouldReplace = 0;
            } else {
	            shouldReplace = 1;
    	    }
	    }
	}

    if (shouldReplace) {
        if (!addr->evalType) {
            ttFilledSize++;
        }

        *addr = *newEntry;
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

Transposition* getTTEntry(U64 key) {
    Transposition* transposition = &tt[key & ttIndex];

    return transposition;
}