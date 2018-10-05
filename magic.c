#include "magic.h"

void magicGen() {
    //Генерация для ладьи
    int max = 0;
    for(int sq = 0; sq < 64; ++sq) {
        rookMagicMask[sq] = (plus1[sq] & ~files[7])
                          | (minus1[sq] & ~files[0])
                          | (minus8[sq] & ~ranks[0])
                          | (plus8[sq] & ~ranks[7]);

        int cur_attacks_cnt = popcount(rookMagicMask[sq]);
        max = (cur_attacks_cnt > max ? cur_attacks_cnt : max);
        
        U64 magic = magicFind(rookMagicMask[sq]);
        if(sq % 8 == 0) {
            printf("\n");
        }
        printf("0x%lx, ", magic);
    }
    printf("Max for rooks: %d\n", max);

    //Генерация для слона
    max = 0;
    for(int sq = 0; sq < 64; ++sq) {
        bishopMagicMask[sq] = (plus9[sq] & ~(files[7] | ranks[7]))
                            | (plus7[sq] & ~(files[0] | ranks[7]))
                            | (minus9[sq] & ~(files[0] | ranks[0]))
                            | (minus7[sq] & ~(files[7] | ranks[0]));

        int cur_attacks_cnt = popcount(bishopMagicMask[sq]);
        max = (cur_attacks_cnt > max ? cur_attacks_cnt : max);
        
        U64 magic = magicFind(rookMagicMask[sq]);
        if(sq % 8 == 0) {
            printf("\n");
        }
        printf("0x%lx, ", magic);
    }
    printf("Max for bishops: %d\n", max);
}

U64 getAsIndex(U64 bitboard, int index) {
    U64 result = 0;


    for(int shift = 0; bitboard; ++shift) {
        int sq = ctz(bitboard);

        if((1ull << shift) & index) {
            setBit(&result, sq);
        }

        clearBit(&bitboard, sq);
    }

    return result;
}

U64 magicFind(U64 bitboard) {
    U64 magic;
    
    while(1) {
        magic = magicRand();

        if(perfectHashTest(bitboard, magic)) {
            return magic;
        }        
    }

    return magic;
}

int perfectHashTest(U64 bitboard, U64 magic) {
    int shift = popcount(bitboard) - 1;
    int size = 1 << shift;
    int array[size];

    memset(array, 0, sizeof(int) * size);

    for(int i = 0; i < size; ++i) {
        U64 configuration = getAsIndex(bitboard, i);
        array[(configuration * magic) >> (64 - shift)] = 1;
    }

    for(int i = 0; i < size; ++i) {
        if(!array[i]) {
            return 0;
        }
    }

    return 1;
}

U64 magicRand() {
    U64 result = 0;
    
    for(int i = 0; i < 7; ++i) {
        int shift = rand() % 64;
        result |= (1ull << shift);
    }
    return result;
}