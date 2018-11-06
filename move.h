#ifndef MOVE_H
#define MOVE_H

#include "types.h"
#include "board.h"
#include "movegen.h"

/*
Move:

6 bits -- from
6 bits - to
2 bits - move type
2 bits - promo type
*/

//Move type
#define NORMAL_MOVE (0 << 12)
#define ENPASSANT_MOVE (1 << 12)
#define CASTLE_MOVE (2 << 12)
#define PROMOTION_MOVE (3 << 12)

//Promo type
#define PROMOTE_TO_KNIGHT (0 << 14)
#define PROMOTE_TO_BISHOP (1 << 14)
#define PROMOTE_TO_ROOK (2 << 14)
#define PROMOTE_TO_QUEEN (3 << 14)

//Promo

#define KNIGHT_PROMOTE_MOVE (PROMOTION_MOVE | PROMOTE_TO_KNIGHT)
#define BISHOP_PROMOTE_MOVE (PROMOTION_MOVE | PROMOTE_TO_BISHOP)
#define ROOK_PROMOTE_MOVE (PROMOTION_MOVE | PROMOTE_TO_ROOK)
#define QUEEN_PROMOTE_MOVE (PROMOTION_MOVE | PROMOTE_TO_QUEEN)

//Move functions
#define MoveFrom(move) (move & 63)
#define MoveTo(move) ((move >> 6) & 63)
#define MoveType(move) (move & (3 << 12))
#define MovePromotionType(move) (move & (3 << 14))
#define MovePromotionPiece(move) (2 + ((move & (3 << 14)) >> 14))
#define MakeMove(from, to, flags) ((from) | ((to) << 6) | (flags))

void moveToString(U16 move, char* str);
U16 stringToMove(Board* board, char* str);

#endif