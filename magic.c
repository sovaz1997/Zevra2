#include "magic.h"

const U64 rookMagic[] = {
    0x880082040028000, 0x8040201000400110, 0x2480300020008000, 0x80080010018401, 0x4100080100100400, 0x1000300180c0000, 0x600180200040100, 0x4080010021408000, 
    0x888000a0401000, 0x41008501400000, 0x4001002000410880, 0x1001001000210108, 0x24800804008020, 0x43010008040010, 0x8000810080120020, 0x1040800500008000, 
    0x80808004400800, 0x10084020004020, 0x840808020100020, 0x10301001020004, 0x8800808008040010, 0x890024010000, 0x40088101201, 0x820000840104, 
    0x4080400080048002, 0x4005000c0200100, 0x200410100200014, 0x10001100210001, 0x404a0200201000, 0x902001200040800, 0x1000460081008000, 0x2010200008404, 
    0x2002400080800040, 0x40201000c0400800, 0x4080060200024, 0x218003000808000, 0x9004804008080000, 0x1000401004821, 0x1020821004000800, 0x405020000e0, 
    0x800090c0008020, 0x2800400100850000, 0x100410020030008, 0x100800868000, 0x8081001008030000, 0x110800402008004, 0x2001080210040004, 0x400800100428020, 
    0x420088020400080, 0x10009040200080, 0x2004120028200, 0x480100040220200, 0x302801040100, 0x1004a0400802, 0x1001200800d00, 0x1040281001200, 
    0x12004021008002, 0x240040820104, 0x200012004482, 0x8210088001001, 0x40100801000401, 0x4001008200080401, 0x803000200048001, 0x800404100008402
};

const U64 bishopMagic[] = { 
    0x80044000208204, 0x9040100020004004, 0x1080300020008040, 0x880080050008200, 0x2000e0010082000, 0x80140002018000, 0x900120024010000, 0x100004082010012, 
    0x8000a0400101, 0x1004001810090, 0x2004020920400, 0x100a510010000, 0x800804008301, 0x1401000804010000, 0x9201010004020000, 0x4088800100208000, 
    0x4102808000400200, 0x20204001500200, 0x1050020004400, 0x1420012220000, 0x21000a0022001000, 0x2001030004000810, 0x1040002081001, 0x400060004012080, 
    0x200400080008448, 0x220204840100000, 0x5000430100200000, 0x1008008480100100, 0x44008280080200, 0x10089000c0000, 0x81400231000, 0x14010200042080, 
    0x40400090800108, 0x21c0081000200000, 0x1001002001004011, 0xc802010010100000, 0x400080101001003, 0x2040180401010000, 0x480204001004, 0x80040082000b00, 
    0x44400080008006, 0xa010080c0010000, 0x244402001010000, 0x290001088008000, 0x501801010000, 0x3000884010008, 0x20110802040000, 0x8001000a8002, 
    0x101008012400100, 0x2804020010500, 0x10008800200380, 0x30008018040080, 0x2008108040080, 0x4010600c0800, 0x4000802201004080, 0x2080804001000080, 
    0x8800040012101, 0x200820100400022, 0x2001024020001001, 0x201100100080021, 0x1000400101801, 0x8000021040200408, 0x200080200100304, 0x1000410000440082
};

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