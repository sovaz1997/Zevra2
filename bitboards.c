#include "bitboards.h"

void initBitboards() {
    for(int i = 0; i < 8; ++i) {
        ranks[i] = (255ull << (i * 8));
        files[i] = (72340172838076673ull << i);
    }

    shortCastlingBitboard[WHITE] = bitboardCell(square(0, 4)) | bitboardCell(square(0, 7));
    longCastlingBitboard[WHITE] = bitboardCell(square(0, 4)) | bitboardCell(square(0, 0));
    shortCastlingBitboard[BLACK] = bitboardCell(square(7, 4)) | bitboardCell(square(7, 7));
    longCastlingBitboard[BLACK] = bitboardCell(square(7, 4)) | bitboardCell(square(7, 0));

    //Инициализация лучей
    memset(plus1, 0, sizeof(U64) * 64);
    memset(plus7, 0, sizeof(U64) * 64);
    memset(plus8, 0, sizeof(U64) * 64);
    memset(plus9, 0, sizeof(U64) * 64);
    memset(minus1, 0, sizeof(U64) * 64);
    memset(minus7, 0, sizeof(U64) * 64);
    memset(minus8, 0, sizeof(U64) * 64);
    memset(minus9, 0, sizeof(U64) * 64);
    memset(squareBitboard, 0, sizeof(U64) * 64);
    memset(unSquareBitboard, 0, sizeof(U64) * 64);

    for(int sq = 0; sq < 64; ++sq) {
        squareBitboard[sq] = bitboardCell(sq);
        unSquareBitboard[sq] = ~squareBitboard[sq];
    }

    for(int sq = 0; sq < 64; ++sq) {
        for(int r = rankOf(sq) + 1, f = fileOf(sq) + 1; r < 8 && f < 8; ++r, ++f) {
            setBit(&plus9[sq], square(r, f));
        }
        for(int r = rankOf(sq) + 1, f = fileOf(sq) - 1; r < 8 && f >= 0; ++r, --f) {
            setBit(&plus7[sq], square(r, f));
        }
        for(int r = rankOf(sq) - 1, f = fileOf(sq) - 1; r >= 0 && f >= 0; --r, --f) {
            setBit(&minus9[sq], square(r, f));
        }
        for(int r = rankOf(sq) - 1, f = fileOf(sq) + 1; r >= 0 && f < 8; --r, ++f) {
            setBit(&minus7[sq], square(r, f));
        }

        for(int r = rankOf(sq) + 1; r < 8; ++r) {
            setBit(&plus8[sq], square(r, fileOf(sq)));
        }
        for(int r = rankOf(sq) - 1; r >= 0; --r) {
            setBit(&minus8[sq], square(r, fileOf(sq)));
        }
        for(int f = fileOf(sq) + 1; f < 8; ++f) {
            setBit(&plus1[sq], square(rankOf(sq), f));
        }
        for(int f = fileOf(sq) - 1; f >= 0; --f) {
            setBit(&minus1[sq], square(rankOf(sq), f));
        }
    }

    attacksGen();
}

void attacksGen() {
    memset(knightAttacks, 0, sizeof(U64) * 64);
    memset(pawnMoves, 0, sizeof(U64) * 64 * 2);

    //Генерация атак ладьи
    for(int sq = 0; sq < 64; ++sq) {
        rookAttacks[sq] = (minus1[sq] | plus1[sq] | minus8[sq] | plus8[sq]);
    }

    //Генерация атак слона
    for(int sq = 0; sq < 64; ++sq) {
        bishopAttacks[sq] = (minus7[sq] | plus7[sq] | minus9[sq] | plus9[sq]);
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

    //Генерация атак короля

    for(int sq = 0; sq < 64; ++sq) {
        int r = rankOf(sq);
        int f = fileOf(sq);

        if(r + 1 < 8 && f + 1 < 8) {
            setBit(&kingAttacks[sq], square(r + 1, f + 1));
        }
        if(r + 1 < 8 && f - 1 >= 0) {
            setBit(&kingAttacks[sq], square(r + 1, f - 1));
        }
        if(r + 1 < 8) {
            setBit(&kingAttacks[sq], square(r + 1, f));
        }

        if(r - 1 >= 0 && f + 1 < 8) {
            setBit(&kingAttacks[sq], square(r - 1, f + 1));
        }
        if(r - 1 >= 0 && f - 1 >= 0) {
            setBit(&kingAttacks[sq], square(r - 1, f - 1));
        }
        if(r - 1 >= 0) {
            setBit(&kingAttacks[sq], square(r - 1, f));
        }

        if(f + 1 < 8) {
            setBit(&kingAttacks[sq], square(r, f + 1));
        }
        if(f - 1 >= 0) {
            setBit(&kingAttacks[sq], square(r, f - 1));
        }
    }

    //Генерация ходов пешки
    for(int i = square(1, 0); i < square(7, 0); ++i) {
        if(rankOf(i) == 1) {
            setBit(&pawnMoves[WHITE][i], i + 8);
            setBit(&pawnMoves[WHITE][i], i + 16);
        } else {
            setBit(&pawnMoves[WHITE][i], i + 8);
        }        
    }
    for(int i = square(6, 7); i >= square(1, 0); --i) {
        if(rankOf(i) == 6) {
            setBit(&pawnMoves[BLACK][i], i - 8);
            setBit(&pawnMoves[BLACK][i], i - 16);
        } else {
            setBit(&pawnMoves[BLACK][i], i - 8);
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

unsigned int clz(U64 bitboard) {
    return __builtin_clzll(bitboard);
}

unsigned int firstOne(U64 bitboard) {
    return ctz(bitboard);
}

unsigned int lastOne(U64 bitboard) {
    return 63 - clz(bitboard);
}

unsigned int ctz(U64 bitboard) {
    
    return __builtin_ctzll(bitboard);
}

void setBit(uint64_t* bitboard, int sq) {
    (*bitboard) |= squareBitboard[sq];
}

int getBit(uint64_t bitboard, int sq) {
    return !!(bitboard & squareBitboard[sq]);
}

void clearBit(uint64_t* bitboard, int sq) {
    (*bitboard) &= unSquareBitboard[sq];
}

int rankOf(int sq) {
    return sq / 8;
}

int fileOf(int sq) {
    return sq % 8;
}