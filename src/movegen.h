#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "types.h"
#include "board.h"
#include "move.h"
#include "magic.h"

void movegen(Board* board, uint16_t* moveList);
void attackgen(Board* board, uint16_t* moveList);
uint16_t* genMovesFromBitboard(int from, U64 bitboard, uint16_t* moveList);
uint16_t* genPromoMovesFromBitboard(int from, U64 bitboard, uint16_t* moveList);
uint16_t* genPawnCaptures(U64 to, int diff, uint16_t* moveList, U16 flags);

#endif