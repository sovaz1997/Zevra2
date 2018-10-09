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
    0x10100088008021, 0x4008102400802000, 0x10018200401008, 0x4042280000240, 0x402021000000404, 0x4002080208000002, 0x858820100000, 0x10140101101000,
    0x4000101001410400, 0x80888008420, 0x80041404004010, 0x40080841001800, 0x20211400010, 0x1020806084000, 0x800020804048400, 0x42002108021000,
    0x20008820110200, 0x2200910040080, 0x1001004002144, 0x48002082004000, 0x42c000080a00000, 0x1000601010108, 0x1400401041020, 0x408212008400,
    0x804200090200100, 0x21080004080840, 0x680010084040, 0x88080000202020, 0x21005001014000, 0x1208002002008400, 0x1020004088410, 0x2000404000840410,
    0x8045000420200, 0x1082000420420, 0x2008205100020, 0x828080180200, 0x4008404040040100, 0xa0008080010802, 0x4040402008090, 0x812100104400,
    0x1210820004080, 0x2080402202400, 0x82888001000, 0x10101148000400, 0x20080104000640, 0x1901102000040, 0x200800c4800100, 0x4008010400808020,
    0xc010148200000, 0x9004804040004, 0x221052080000, 0x8001000042020410, 0x1000801016020000, 0x8000062004010000, 0x20040108012800, 0x10020604012000,
    0x410800900420, 0x80004404048200, 0x400000184008810, 0x200020001840401, 0x102001010420200, 0x1002004504080, 0x80808880050, 0x202041000820080
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
        
        U64 magic = magicFind(bishopMagicMask[sq]);
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

            
            U64* possibleMoves = &rookPossibleMoves[sq][getMagicIndex(occu, rookMagic[sq], rookPossibleMovesSize[sq])];
            *possibleMoves = blockerCut(sq, occu, plus8, UP, *possibleMoves);
            *possibleMoves = blockerCut(sq, occu, plus1, UP, *possibleMoves);
            *possibleMoves = blockerCut(sq, occu, minus8, DOWN, *possibleMoves);
            *possibleMoves = blockerCut(sq, occu, minus1, DOWN, *possibleMoves);

            occu = getAsIndex(bishopMagicMask[sq], i);

            possibleMoves = &bishopPossibleMoves[sq][getMagicIndex(occu, bishopMagic[sq], bishopPossibleMovesSize[sq])];
            *possibleMoves = blockerCut(sq, occu, plus9, UP, *possibleMoves);
            *possibleMoves = blockerCut(sq, occu, plus7, UP, *possibleMoves);
            *possibleMoves = blockerCut(sq, occu, minus9, DOWN, *possibleMoves);
            *possibleMoves = blockerCut(sq, occu, minus7, DOWN, *possibleMoves);        
        }
    }
}

U64 blockerCut(int from, U64 occu, U64* directionArray, int direction, U64 possibleMoves) {
    U64 directAttack = occu & directionArray[from];
    possibleMoves |= directionArray[from];
    if(popcount(directAttack)) {
        int blocker = ((direction == UP) ? ctz(directAttack) : 63 - clz(directAttack));
        return possibleMoves ^ directionArray[blocker];
    }
    return possibleMoves;
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