#include "move.h"

void moveToString(uint16_t move, char* str) {
    squareToString(MoveFrom(move), str);
    squareToString(MoveTo(move), str + 2);
    str[4] = '\0';

    if(MoveType(move) == PROMOTION_MOVE) {
        str[4] = PieceName[BLACK][MovePromotionPiece(move)];
        str[5] = '\0';
    }
}

U16 stringToMove(Board* board, char* str) {
    U16 moveList[256];
    movegen(board, moveList);

    U16* ptr = moveList;
    while(*ptr) {
        char cmp_str[6];
        moveToString(*ptr, cmp_str);
        if(!strcmp(cmp_str, str))
            return *ptr;
        ++ptr;
    }

    return 0;
}

void printMove(U16 move) {
    char moveStr[6];
    moveToString(move, moveStr);
    printf("%s\n", moveStr);
}

char* getMove(U16 move) {
    char* moveStr = malloc(sizeof(char) * 6);
    moveToString(move, moveStr);
    return moveStr;
}