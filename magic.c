#include "magic.h"

const U64 rookMagic[] = {
    36031546336002048ull, 1170935971854680384ull, 108121575697418368ull, 36046391357474816ull, 612496163572547712ull, 144123992759866369ull, 180216553147465856ull, 2341872355990716672ull, 
    576601491942457344ull, 141012510965888ull, 9367628031146926080ull, 288371664066973696ull, 422229653329920ull, 562984582774804ull, 288511859718652160ull, 4611826757006262912ull, 
    2305993092588634144ull, 4504974155325456ull, 580964489437380768ull, 282574756776097ull, 9225624387028287496ull, 72340168664810496ull, 16140902168302583812ull, 1128098997227649ull, 
    1196275095732224ull, 36063986760812608ull, 35188671316176ull, 18296290099068936ull, 4612248994151663648ull, 4647719215640739968ull, 11261200247554304ull, 9223512776490770688ull, 
    4611756936944165632ull, 1267874354184192ull, 5084178282455040ull, 2378041409467844608ull, 9223943785057096704ull, 2666133180582462464ull, 144537409139249408ull, 2305983756374114560ull, 
    1306184630499573792ull, 720611125825142912ull, 281784751292433ull, 40541261475676192ull, 144141576422064256ull, 22265647857936ull, 306246973701259392ull, 285875175030793ull, 
    10412322613368979520ull, 4612284154904578176ull, 144132782478590592ull, 38579666143412352ull, 27303107100938496ull, 37154715213004928ull, 18015532649423872ull, 140741783606400ull, 
    36310410516758545ull, 2306124553450946689ull, 72076286272473153ull, 4436703350785ull, 72620612780034050ull, 2533291970398209ull, 584166019076ull, 4785076785127489ull
};

const U64 bishopMagic[] = { 
    1127025222091008ull, 9572631766433792ull, 4539892103120897ull, 9224572705708179456ull, 72356936347025472ull, 18174997000355904ull, 18304706153807872ull, 141321612435584ull, 
    149602368489536ull, 145152723387648ull, 4591715180625920ull, 9043621659541504ull, 567417407275072ull, 1152922621835485312ull, 1152922064162653184ull, 9511602971486520320ull, 
    1125977418106880ull, 580964360587904008ull, 4611967502128382082ull, 1125934870573056ull, 1442277817183109120ull, 281481966522368ull, 571755727167488ull, 1266646018720384ull, 
    74379765280407808ull, 581367310319744ull, 1161651483115584ull, 3940787247153184ull, 4611969211407810560ull, 4683884899726794752ull, 4611986185169142016ull, 1126999487717888ull, 
    1161118640800260ull, 1284538819151872ull, 144187895430775040ull, 72079588699930880ull, 1157427311847432200ull, 4521210100613184ull, 563525479305728ull, 4612253367501074944ull, 
    2306564306558259200ull, 303469588127744ull, 107752944828928ull, 9223935270284429328ull, 8800392252416ull, 18016599714300160ull, 1127162660783104ull, 10141897452421152ull, 
    3378805810921472ull, 3458837090196783104ull, 585325084672ull, 562951572946948ull, 2310346678131523584ull, 72092918063611904ull, 289391606493216768ull, 4620694321505568768ull, 
    70523380318336ull, 1104352912384ull, 576461852360577152ull, 35321844925456ull, 149181738462085632ull, 279323934980ull, 17660939600066ull, 4504703451136512ull
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
        printf("%llu, ", magic);
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
        printf("%llu, ", magic);
    }
    printf("Max for bishops: %d\n", max);
}

U64 getAsIndex(U64 bitboard, int index) {
    U64 result = 0;


    for(int shift = 0; bitboard; ++shift) {
        int sq = firstOne(bitboard);

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
    const int size = 1 << popcount(bitboard);
    
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
        
        int array[1 << popcount(bitboard)];
        memset(array, 0, (1 << popcount(bitboard)) * sizeof(int));
        for(int i = 0; i < (1 << (popcount(bitboard))); ++i) {
            U64 occu = getAsIndex(rookMagicMask[sq], i);

            U64* possibleMoves = &rookPossibleMoves[sq][getMagicIndex(occu, rookMagic[sq], rookPossibleMovesSize[sq])];
            *possibleMoves = blockerCut(sq, occu, plus8, UP, *possibleMoves);
            *possibleMoves = blockerCut(sq, occu, plus1, UP, *possibleMoves);
            *possibleMoves = blockerCut(sq, occu, minus8, DOWN, *possibleMoves);
            *possibleMoves = blockerCut(sq, occu, minus1, DOWN, *possibleMoves);   
        }

        bitboard = bishopMagicMask[sq];

        for(int i = 0; i < (1 << (popcount(bitboard))); ++i) {
            U64 occu = getAsIndex(bishopMagicMask[sq], i);
            U64* possibleMoves = &bishopPossibleMoves[sq][getMagicIndex(occu, bishopMagic[sq], bishopPossibleMovesSize[sq])];
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
        int blocker = ((direction == UP) ? firstOne(directAttack) : lastOne(directAttack));
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