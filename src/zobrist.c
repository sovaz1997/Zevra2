#include "zobrist.h"

U64 nextSeed = 0x8a432df075f3;

U64 rand64() {
	//U64 x = state;	
	nextSeed ^= nextSeed >> 12;
	nextSeed ^= nextSeed << 25;
	nextSeed ^= nextSeed >> 27;
    nextSeed *= 0x2545F4914F6CDD1D;
	return nextSeed;
}

void zobristInit() {
    for(int i = 0; i < 15; ++i) {
        for(int j = 0; j < 64; ++j) {
            zobristKeys[i][j] = rand64();
        }
    }

    for(int i = 0; i < 4; ++i) {
        zobristCastlingKeys[i] = rand64();
    }

    for(int i = 0; i < 64; ++i) {
        zobristEnpassantKeys[i] = rand64();
    }

    nullMoveKey = rand64();
    otherSideKey = rand64();
}