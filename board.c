#include "board.h"

void setFen(Board* board, char* fen) {
    clearBoard(board);

    char* str = strdup(fen);
    char* savePtr = NULL;

    char* pieces_str = strtok_r(str, " ", &savePtr);
    

    char* color_str = strtok_r(NULL, " ", &savePtr);
    char* castlings_str = strtok_r(NULL, " ", &savePtr);
    char* enpassantSquare_str = strtok_r(NULL, " ", &savePtr);
    char* ruleNumber_str = strtok_r(NULL, " ", &savePtr);
    char* moveNumber_str = strtok_r(NULL, " ", &savePtr);

    //Установка фигур

    int f = 0, r = 7;

    while(*pieces_str) {
        if(isdigit(*pieces_str)) {
           f += (*pieces_str - '0');
        } else if(*pieces_str == '/') {
            --r;
            f = 0;
        } else {
            int color = !!islower(*pieces_str);
            char* piece = strchr(PieceName[color], *pieces_str);
            setPiece(board, piece - PieceName[color], color, square(r, f++));
        }

        ++pieces_str;
    }

    board->color = (*color_str == 'w' ? WHITE : BLACK);

    //Установка рокировки

    while(*castlings_str) {
        if(*castlings_str == 'K') {
            board->castling |= WS_CASTLING;
        } else if(*castlings_str == 'Q') {
            board->castling |= WL_CASTLING;
        } else if(*castlings_str == 'k') {
            board->castling |= BL_CASTLING;
        } else if(*castlings_str == 'q') {
            board->castling |= BL_CASTLING;
        }

        ++castlings_str;
    }

    //Установка поля взятия на проходе
    board->enpassantSquare = (*enpassantSquare_str == '-' ? -1 : stringToSquare(enpassantSquare_str));
    
    //Установка номера ходв без взятий и проведений пешек
    board->ruleNumber = atoi(ruleNumber_str);
    
    //Установка номера хода
    board->moveNumber = atoi(moveNumber_str);
    free(str);
}

void getFen(Board* board, char* fen) {

}

void clearBoard(Board* board) {
    memset(board, 0, sizeof(*board));
    board->enpassantSquare = -1;
}

void printBoard(Board* board) {
    printBoardSplitter();

    for(int r = 7; r >= 0; --r) {
        printf("| ");
        for(int f = 0; f < 8; ++f) {
            printPiece(board->squares[square(r, f)]);
            printf(" | ");
        }
        printf("\n");
        printBoardSplitter();
    }
}

void printBoardSplitter() {
    for(int i = 0; i < 33; ++i) {
        if(i % 4 == 0) {
            printf("+");
        } else {
            printf("-");
        }
    }
    printf("\n");
}

void setPiece(Board* board, int piece, int color, int square) {
    clearPiece(board, square);
    setBit(&board->pieces[piece], square);
    setBit(&board->colours[color], square);
    board->squares[square] = makePiece(piece, color);
}

void clearPiece(Board* board, int square) {
    U8 piece = board->squares[square];
    clearBit(&board->pieces[pieceType(piece)], square);
    clearBit(&board->colours[pieceColor(piece)], square);
    board->squares[square] = 0;
}

void movePiece(Board* board, int sq1, int sq2) {
    setPiece(board, pieceType(board->squares[sq1]), pieceColor(board->squares[sq1]), sq2);
    clearPiece(board, sq1);
}

void squareToString(int square, char* str) {
    str[0] = fileOf(square) + 'a';
    str[1] = rankOf(square) + '1';
    str[2] = '\0';
}

int stringToSquare(char* str) {
    return ((str[0] - 'a') + (str[1] - '1') * 8);
}

U8 makePiece(int piece_type, int color) {
    return (piece_type << 1) + color;
}

void printPiece(U8 piece) {
    if(piece) {
        printf("%c", PieceName[pieceColor(piece)][pieceType(piece)]);
    } else {
        printf(" ");
    }
}

void makeMove(Board* board, U16 move, Undo* undo) {
    setUndo(board, undo, board->squares[MoveTo(move)]);
    if(MoveType(move) == NORMAL_MOVE) {
        movePiece(board, MoveFrom(move), MoveTo(move));
    }
    board->color = !board->color;
    ++board->moveNumber;
}

void unmakeMove(Board* board, U16 move, Undo* undo) {
    getUndo(board, undo);
    if(MoveType(move) == NORMAL_MOVE) {
        movePiece(board, MoveTo(move), MoveFrom(move));
        if(undo->capturedPiece) {
            setPiece(board, pieceType(undo->capturedPiece), pieceColor(undo->capturedPiece), MoveTo(move));
        }
    }
    board->color = !board->color;
    --board->moveNumber;
}

void setUndo(Board* board, Undo* undo, U8 capturedPiece) {
    undo->capturedPiece = capturedPiece;
    undo->castling = board->castling;
    undo->enpassantSquare = board->enpassantSquare;
    undo->ruleNumber = board->ruleNumber;
}

void getUndo(Board* board, Undo* undo) {
    board->castling = undo->castling;
    board->ruleNumber = undo->ruleNumber;
    board->castling = undo->castling;
}

int isEqual(Board* b1, Board* b2) {
    if(b1->moveNumber != b2->moveNumber) {
        printf("moveNumber\n");
        return 0;
    }
    if(b1->ruleNumber != b2->ruleNumber) {
        printf("ruleNumber\n");
        return 0;
    }
    if(b1->color != b2->color) {
        printf("ruleNumber\n");
        return 0;
    }
    if(b1->enpassantSquare != b2->enpassantSquare) {
        printf("ruleNumber\n");
        return 0;
    }
    if(b1->castling != b2->castling) {
        printf("castling\n");
        return 0;
    }
    for(int i = 1; i < 7; ++i) {
        if(b1->pieces[i] != b2->pieces[i]) {
            return 0;
        }
    }
    for(int i = 0; i < 2; ++i) {
        if(b1->colours[i] != b2->colours[i]) {
            return 0;
        }
    }
    for(int i = 0; i < 64; ++i) {
        if(b1->squares[i] != b2->squares[i]) {
            return 0;
        }
    }
    return 1;

}

int attackedSquare(Board* board, int sq, int color) {
    U8 attackPiece = firstAttacker(board, plus8[sq]);
    if(attackPiece == makePiece(QUEEN, !color) || attackPiece == makePiece(ROOK, !color)) {
        return 1;
    }

    attackPiece = firstAttacker(board, plus1[sq]);
    if(attackPiece == makePiece(QUEEN, !color) || attackPiece == makePiece(ROOK, !color)) {
        return 1;
    }

    attackPiece = firstAttacker(board, plus7[sq]);
    if(attackPiece == makePiece(QUEEN, !color) || attackPiece == makePiece(BISHOP, !color)) {
        return 1;
    }

    attackPiece = firstAttacker(board, plus9[sq]);
    if(attackPiece == makePiece(QUEEN, !color) || attackPiece == makePiece(BISHOP, !color)) {
        return 1;
    }

    attackPiece = lastAttacker(board, minus8[sq]);
    if(attackPiece == makePiece(QUEEN, !color) || attackPiece == makePiece(ROOK, !color)) {
        return 1;
    }

    attackPiece = lastAttacker(board, minus1[sq]);
    if(attackPiece == makePiece(QUEEN, !color) || attackPiece == makePiece(ROOK, !color)) {
        return 1;
    }

    attackPiece = lastAttacker(board, minus7[sq]);
    if(attackPiece == makePiece(QUEEN, !color) || attackPiece == makePiece(BISHOP, !color)) {
        return 1;
    }

    attackPiece = lastAttacker(board, minus9[sq]);
    if(attackPiece == makePiece(QUEEN, !color) || attackPiece == makePiece(BISHOP, !color)) {
        return 1;
    }

    U64 enemyPawns = board->colours[!color] & board->pieces[PAWN];
    if(color == WHITE) {
        U64 attackedSquares = ((enemyPawns << 9) & ~files[7]) | ((enemyPawns << 7) & ~files[0]);
        if(attackedSquares & bitboardCell(sq)) {
            return 1;
        }
    } else {
        U64 attackedSquares = ((enemyPawns >> 9) & ~files[0]) | ((enemyPawns >> 7) & ~files[7]);
        if(attackedSquares & bitboardCell(sq)) {
            return 1;
        }
    }

    if(kingAttacks[firstOne(board->colours[!color] & board->pieces[KING])] & bitboardCell(sq)) {
        return 1;
    }

    return 0;
}

int inCheck(Board* board, int color) {
    int kingPosition = firstOne(board->colours[color] & board->pieces[KING]);
    return attackedSquare(board, kingPosition, color);
}

U8 firstAttacker(Board* board, U64 bitboard) {
    if(!bitboard) {
        return 0;
    }

    return board->squares[firstOne(bitboard)];
}

U8 lastAttacker(Board* board, U64 bitboard) {
    if(!bitboard) {
        return 0;
    }

    return board->squares[lastOne(bitboard)];
}