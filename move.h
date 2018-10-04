#ifndef MOVE_H
#define MOVE_H

#include "types.h"
#include "board.h"

/*
Move - ход, описывается 16 байтами:

6 бита - откуда
6 бита - куда
2 бита - тип хода
2 бита - тип превращения
*/

//Тип хода
#define NORMAL_MOVE (0 << 12)
#define ENPASSANT_MOVE (1 << 12)
#define CASTLE_MOVE (2 << 12)
#define PROMOTION_MOVE (3 << 12)

//Тип превращения
#define PROMOTE_TO_KNIGHT (0 << 14)
#define PROMOTE_TO_BISHOP (1 << 14)
#define PROMOTE_TO_ROOK (2 << 14)
#define PROMOTE_TO_QUEEN (3 << 14)

//Превращение в фигуру

#define KNIGHT_PROMOTE_MOVE (PROMOTION_MOVE | PROMOTE_TO_KNIGHT)
#define BISHOP_PROMOTE_MOVE (PROMOTION_MOVE | PROMOTE_TO_BISHOP)
#define ROOK_PROMOTE_MOVE (PROMOTION_MOVE | PROMOTE_TO_ROOK)
#define QUEEN_PROMOTE_MOVE (PROMOTION_MOVE | PROMOTE_TO_QUEEN)

//Ход
#define MoveFrom(move) (move & 63)
#define MoveTo(move) ((move >> 6) & 63)
#define MoveType(move) (move & (3 << 12))
#define MovePromotionType(move) (move & (3 << 14))
#define MovePromotionPiece(move) (1 + (move & (3 << 14)))

//Функции
void moveToString(U16 move, char* str);

#endif