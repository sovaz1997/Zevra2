#include "magic.h"

const U64 rookMagic[] = {
    0x2080008018204000, 0x1840004010002000, 0x4100090040102000, 0x80043000802800, 0x2100040300100800, 0x200050200041008, 0x100008412000100, 0x100010000402882,
    0x80800890204000, 0x1010401000200240, 0x8040802000300080, 0x1001004082101, 0x20800800440280, 0x1101000300040008, 0x10800500800200, 0x8010800100204080,
    0x980004000502000, 0x301000c000200040, 0x441010020001040, 0xa0808008011000, 0x2c00808004000800, 0x2042008004008002, 0x840008100209, 0xa0020000804104,
    0x4400080002184, 0x2002200280400080, 0x400100480200080, 0x8000210100085000, 0x8200050100100800, 0x2a0080440080, 0x902000200040801, 0x2000102000080c4,
    0x1200400180800020, 0x810a01000404000, 0x40a0802000801000, 0x11041001002008, 0x8040180804800, 0xc0080804a00, 0x805200800100, 0x1004082000403,
    0x800400082308000, 0x21018040010020, 0x1022000110040, 0x1022010010008, 0x100a80004008080, 0x4082000410020008, 0x200010210140008, 0x1000050810002,
    0x1042040800011, 0x400100802091, 0x1004020001409, 0x2000804401022, 0x1001004220801, 0x1000806240001, 0x4000080201100084, 0x80010080440022,
};

const U64 bishopMagic[] = { 
    0x1080001080400820, 0x8140002000300040, 0x4080100080082001, 0x500100020090004, 0x280030800800400, 0x1180018004000200, 0x680010001800200, 0x100004422008100,
    0x100800220804008, 0x4001c00040201000, 0x1820802000801000, 0x1200801000810800, 0x4210800400080080, 0x2400808004000200, 0x1808051000200, 0x801280024100,
    0x1080004000d02000, 0x400404010012000, 0xc020008020801000, 0x4818008009000, 0x50018001101, 0x400818002000400, 0x2000040010090802, 0x3000020000408104,
    0x802400080208008, 0x1005001c0002000, 0x20008880203000, 0x8000900080080180, 0x200080180040080, 0x4000100801402004, 0x81000900220004, 0x8800040200008041,
    0x800080c1002100, 0x20008100a1004000, 0x820042005020, 0x8020811000800800, 0x102820800800400, 0x800220080800400, 0x401000409000a00, 0x20088000c0800100,
    0x1880084020004000, 0x4210200040008080, 0x200100410011, 0x400090110010020, 0x2140008008080, 0x84008022008004, 0x404010002008080, 0x210010080420004,
    0x100400520800080, 0x400201002400440, 0x2200200410008080, 0x2101000820100100, 0x104000802048080, 0x400800a000480, 0x3801000200040100, 0x8001002040820100,
    0x104280010021, 0x2008100401022, 0x8a000410011, 0x201900100005, 0x200080420100a, 0x4001000400020801, 0x8302100804, 0x1000600802041
};

void magicGen() {
    preInit();

    //Генерация для ладьи
    int max = 0;
    for(int sq = 0; sq < 64; ++sq) {
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
    int size = 1 << popcount(bitboard);
    
    int array[size];

    memset(array, 0, sizeof(int) * size);

    for(int i = 0; i < size; ++i) {
        U64 configuration = getAsIndex(bitboard, i);
        array[getMagicIndex(configuration, magic, popcount(bitboard))] = 1;
    }

    for(int i = 0; i < size; ++i) {
        if(!array[i]) {
            return 0;
        }
    }
    

    return 1;
}

int getMagicIndex(U64 configuration, U64 magic, int size) {
    U64 tmp;
  
    return (configuration * magic) >> (64 - size);
}

U64 magicRand() {
    U64 result = 0;
    
    for(int i = 0; i < 7; ++i) {
        int shift = rand() % 64;
        result |= (1ull << shift);
    }

    return result;
}

void magicArraysInit() {
    preInit();

    for(int sq = 0; sq < 64; ++sq) {
        U64 bitboard = rookMagicMask[sq];
        for(int i = 0; i < (1 << (popcount(bitboard))); ++i) {
            U64 occu = getAsIndex(rookMagicMask[sq], i);


            char str[6];
            squareToString(sq, str);
            //printf("%s:\n", str);
            //printf("%lld\n", occu);
            
            U64 directAttack = plus8[sq] & occu;
            rookPossibleMoves[sq][getMagicIndex(occu, rookMagic[sq], rookPossibleMovesSize[sq])] |= plus8[sq];
            if(popcount(directAttack)) {
                int blocker = ctz(directAttack);
                rookPossibleMoves[sq][getMagicIndex(occu, rookMagic[sq], rookPossibleMovesSize[sq])] ^= plus8[blocker];
            }
            directAttack = plus1[sq] & occu;
            rookPossibleMoves[sq][getMagicIndex(occu, rookMagic[sq], rookPossibleMovesSize[sq])] |= plus1[sq];
            if(popcount(directAttack)) {
                int blocker = ctz(directAttack);
                rookPossibleMoves[sq][getMagicIndex(occu, rookMagic[sq], rookPossibleMovesSize[sq])] ^= plus1[blocker];
            }

            directAttack = minus8[sq] & occu;
            rookPossibleMoves[sq][getMagicIndex(occu, rookMagic[sq], rookPossibleMovesSize[sq])] |= minus8[sq];
            if(popcount(directAttack)) {
                int blocker = 63 - clz(directAttack);
                rookPossibleMoves[sq][getMagicIndex(occu, rookMagic[sq], rookPossibleMovesSize[sq])] ^= minus8[blocker];
            }

            directAttack = minus1[sq] & occu;
            rookPossibleMoves[sq][getMagicIndex(occu, rookMagic[sq], rookPossibleMovesSize[sq])] |= minus1[sq];
            if(popcount(directAttack)) {
                int blocker = 63 - clz(directAttack);
                rookPossibleMoves[sq][getMagicIndex(occu, rookMagic[sq], rookPossibleMovesSize[sq])] ^= minus1[blocker];
            }

            //printf("\n");
            //if(occu == 281474976710782) {
                //printf("%d\n", rookPossibleMovesSize[sq]);
                //printBitboard(rookPossibleMoves[sq][getMagicIndex(occu, rookMagic[sq], rookPossibleMovesSize[sq])]);
                //printf("%d\n", getMagicIndex(occu, rookMagic[sq], rookPossibleMovesSize[sq]));
            //}
        }
    }
}

void preInit() {
    for(int sq = 0; sq < 64; ++sq) {
        rookMagicMask[sq] = (plus1[sq] & ~files[7])
                          | (minus1[sq] & ~files[0])
                          | (minus8[sq] & ~ranks[0])
                          | (plus8[sq] & ~ranks[7]);
        rookPossibleMovesSize[sq] = popcount(rookMagicMask[sq]);
    }

    for(int sq = 0; sq < 64; ++sq) {
        bishopMagicMask[sq] = (plus9[sq] & ~(files[7] | ranks[7]))
                            | (plus7[sq] & ~(files[0] | ranks[7]))
                            | (minus9[sq] & ~(files[0] | ranks[0]))
                            | (minus7[sq] & ~(files[7] | ranks[0]));
        bishopPossibleMovesSize[sq] = popcount(bishopMagicMask[sq]);
    }
}