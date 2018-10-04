#include "move.h"

void moveToString(uint16_t move, char* str) {
    squareToString(MoveFrom(move), str);
    squareToString(MoveTo(move), str + 2);

    if(MoveType(move) == PROMOTION_MOVE) {
        str[4] = PieceName[BLACK][MovePromotionPiece(move)];
    }
}