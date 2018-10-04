#include "bitboards.h"

void initBitboards() {
    for(int i = 0; i < 8; ++i) {
        ranks[i] = (255ull << (i * 8));
        files[i] = (72340172838076673ull << i);
    }
}

void printBitboard(U64 bitboard) {
    for(int i = 7; i >= 0; --i) {
        for(int j = 0; j < 8; ++j) {
            printf("%d", !!(bitboard & bitboardCell(i * 8 + j)));
        }
        printf("\n");
    }
}

int square(int r, int f) {
    return 8 * r + f;
}