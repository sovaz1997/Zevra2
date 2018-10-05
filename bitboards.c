#include "bitboards.h"

void initBitboards() {
    for(int i = 0; i < 8; ++i) {
        ranks[i] = (255ull << (i * 8));
        files[i] = (72340172838076673ull << i);
    }

    attacksGen();
}

void attacksGen() {
    memset(rookAttacks, 0, sizeof(U64) * 64);
    memset(bishopAttacks, 0, sizeof(U64) * 64);
    memset(knightAttacks, 0, sizeof(U64) * 64);

    //Генерация атак ладьи
    for(int sq = 0; sq < 64; ++sq) {
        rookAttacks[sq] = (ranks[rankOf(sq)] | files[fileOf(sq)]);
        clearBit(&rookAttacks[sq], sq);
        printBitboard(rookAttacks[sq]);
        printf("\n");
    }

    //Генерация атак слона
    for(int sq = 0; sq < 64; ++sq) {
        for(int r = rankOf(sq) + 1, f = fileOf(sq) + 1; r < 8 && f < 8; ++r, ++f) {
            setBit(&bishopAttacks[sq], square(r, f));
        }
        for(int r = rankOf(sq) + 1, f = fileOf(sq) - 1; r < 8 && f >= 0; ++r, --f) {
            setBit(&bishopAttacks[sq], square(r, f));
        }
        for(int r = rankOf(sq) - 1, f = fileOf(sq) + 1; r >= 0 && f < 8; --r, ++f) {
            setBit(&bishopAttacks[sq], square(r, f));
        }
        for(int r = rankOf(sq) - 1, f = fileOf(sq) - 1; r >= 0 && f >= 0; --r, --f) {
            setBit(&bishopAttacks[sq], square(r, f));
        }
    }

    //Генерация атак коня

    for(int sq = 0; sq < 64; ++sq) {
        int r = rankOf(sq);
        int f = fileOf(sq);

        if(r + 2 < 8 && f + 1 < 8) {
            setBit(&knightAttacks[sq], square(r + 2, f + 1));
        }
        if(r + 2 < 8 && f - 1 >= 0) {
            setBit(&knightAttacks[sq], square(r + 2, f - 1));
        }
        if(r - 2 >= 0 && f + 1 < 8) {
            setBit(&knightAttacks[sq], square(r - 2, f + 1));
        }
        if(r - 2 >= 0 && f - 1 >= 0) {
            setBit(&knightAttacks[sq], square(r - 2, f - 1));
        }
        if(r + 1 < 8 && f + 2 < 8) {
            setBit(&knightAttacks[sq], square(r + 1, f + 2));
        }
        if(r + 1 < 8 && f - 2 >= 0) {
            setBit(&knightAttacks[sq], square(r + 1, f - 2));
        }
        if(r - 1 >= 0 && f + 2 < 8) {
            setBit(&knightAttacks[sq], square(r - 1, f + 2));
        }
        if(r - 1 >= 0 && f - 2 >= 0) {
            setBit(&knightAttacks[sq], square(r - 1, f - 2));
        }
    }
}

void printBitboard(U64 bitboard) {
    for(int i = 7; i >= 0; --i) {
        for(int j = 0; j < 8; ++j) {
            printf("%d", !!(bitboard & bitboardCell(square(i, j))));
        }
        printf("\n");
    }
}

unsigned int square(unsigned int r, unsigned int f) {
    return 8 * r + f;
}

unsigned int popcount(U64 bitboard) {
    return __builtin_popcountll(bitboard);
}

unsigned int clz(unsigned int num) {
    return __builtin_clz(num);
}

unsigned int ctz(unsigned int num) {
    return __builtin_ctz(num);
}

void setBit(uint64_t* bitboard, int sq) {
    (*bitboard) |= (1ull << sq);
}

int getBit(uint64_t bitboard, int sq) {
    return !!(bitboard & (1 << sq));
}

int clearBit(uint64_t* bitboard, int sq) {
    (*bitboard) &= ~(1 << sq);
}

int rankOf(int sq) {
    return sq / 8;
}

int fileOf(int sq) {
    return sq % 8;
}