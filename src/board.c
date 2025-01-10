#include "board.h"
#include "uci.h"

void setFen(Board* board, char* fen) {
    clearBoard(board);

    char* str = strdup(fen);

    char* pieces_str = strtok(str, " ");

    char* color_str = strtok(NULL, " ");
    char* castlings_str = strtok(NULL, " ");
    char* enpassantSquare_str = strtok(NULL, " ");
    char* ruleNumber_str = strtok(NULL, " ");
    char* moveNumber_str = strtok(NULL, " ");

    int f = 0, r = 7;

    while(*pieces_str) {
        if(isdigit(*pieces_str)) {
           f += (*pieces_str - '0');
        } else if(*pieces_str == '/') {
            --r;
            f = 0;
        } else {
            int color = !!islower(*pieces_str);
            const char* piece = strchr(PieceName[color], *pieces_str);
            setPiece(board, piece - PieceName[color], color, square(r, f++));
        }

        ++pieces_str;
    }

    board->color = (*color_str == 'w' ? WHITE : BLACK);

    while(*castlings_str) {
        if(*castlings_str == 'K')
            board->castling |= shortCastlingBitboard[WHITE];
        else if(*castlings_str == 'Q')
            board->castling |= longCastlingBitboard[WHITE];
        else if(*castlings_str == 'k')
            board->castling |= shortCastlingBitboard[BLACK];
        else if(*castlings_str == 'q')
            board->castling |= longCastlingBitboard[BLACK];

        ++castlings_str;
    }

    board->enpassantSquare = (*enpassantSquare_str == '-' ? 0 : stringToSquare(enpassantSquare_str));
    board->ruleNumber = atoi(ruleNumber_str);
    board->moveNumber = atoi(moveNumber_str);
    
    free(str);

    if(board->color == BLACK)
        board->key ^= otherSideKey;
}

void setMovesRange(Board* board, char* moves) {
    if(moves) {
        char* move = strtok(moves, " ");
        Undo undo;
        while(move) {
            makeMove(board, stringToMove(board, move), &undo);
            move = strtok(NULL, " ");
        }
    }
}

void clearBoard(Board* board) {
    GameInfo* gameInfo = board->gameInfo;
    memset(board, 0, sizeof(*board));
    board->gameInfo = gameInfo;
    board->enpassantSquare = 0;
    board->eval = 0;

    if (NNUE_ENABLED) {
    	resetNNUE(nnue);
    }
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

    printf("Key: %llu\n\n", board->key);
    printf("White check: %d \n", inCheck(board, WHITE));
    printf("Black check: %d \n\n", inCheck(board, BLACK));
    printf("Have promotion: %d\n", havePromotionPawn(board));
}

void printBoardSplitter() {
    for(int i = 0; i < 33; ++i) {
        if(i % 4 == 0)
            printf("+");
        else
            printf("-");
    }
    printf("\n");
}

void setPiece(Board* board, int piece, int color, int sq) {
    clearPiece(board, sq);
    setBit(&board->pieces[piece], sq);
    setBit(&board->colours[color], sq);
    board->squares[sq] = makePiece(piece, color);
    board->key ^= zobristKeys[board->squares[sq]][sq];

    if (NNUE_ENABLED) {
        setNNUEInput(nnue, getInputIndexOf(color, piece, sq));
    }
}

void clearPiece(Board* board, int sq) {
    if(!board->squares[sq])
        return;
    
    U8 piece = board->squares[sq];
    board->key ^= zobristKeys[board->squares[sq]][sq];

    int type = pieceType(piece);
    int color = pieceColor(piece);
    clearBit(&board->pieces[type], sq);
    clearBit(&board->colours[color], sq);
    board->squares[sq] = 0;

    resetNNUEInput(nnue, getInputIndexOf(color, type, sq));

}

void movePiece(Board* board, int sq1, int sq2) {
    int type = pieceType(board->squares[sq1]);
    int color = pieceColor(board->squares[sq1]);

    clearPiece(board, sq1);
    setPiece(board, type, color, sq2);
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
    if(piece)
        printf("%c", PieceName[pieceColor(piece)][pieceType(piece)]);
    else
        printf(" ");
}

void makeMove(Board* board, U16 move, Undo* undo) {
    setUndo(board, undo, board->squares[MoveTo(move)]);

    if(MoveType(move) == NORMAL_MOVE && !board->squares[MoveTo(move)] && pieceType(board->squares[MoveFrom(move)]) != PAWN)
        ++board->ruleNumber;
    else
        board->ruleNumber = 0;

    movePiece(board, MoveFrom(move), MoveTo(move));
    int epClear = 1;
    if(MoveType(move) == CASTLE_MOVE) {
        U8 king = makePiece(KING, board->color);
        int castlingRank = (board->color == WHITE ? 0 : 7);
        if(board->squares[square(castlingRank, 6)] == king) {
            board->key ^= zobristCastlingKeys[board->color * 2];
            movePiece(board, square(castlingRank, 7), square(castlingRank, 5));
        } else if(board->squares[square(castlingRank, 2)] == king) {
            board->key ^= zobristCastlingKeys[board->color * 2 + 1];
            movePiece(board, square(castlingRank, 0), square(castlingRank, 3));
        }
    } else if(MoveType(move) == PROMOTION_MOVE) {
        setPiece(board, MovePromotionPiece(move), board->color, MoveTo(move));
    } else if(MoveType(move) == ENPASSANT_MOVE) {
        board->key ^= zobristEnpassantKeys[board->enpassantSquare];
        if(board->color == WHITE)
            clearPiece(board, board->enpassantSquare - 8);
        else
            clearPiece(board, board->enpassantSquare + 8);
    } else {
        if(board->squares[MoveTo(move)] == makePiece(PAWN, board->color)) {
            if(abs(MoveTo(move) - MoveFrom(move)) == 16) {
                if(board->color == WHITE)
                    board->enpassantSquare = MoveTo(move) - 8;
                else
                    board->enpassantSquare = MoveTo(move) + 8;
                epClear = 0;
            }
        }
    }

    if(epClear)
        board->enpassantSquare = 0;

    board->color = !board->color;
    
    if(board->color == WHITE)
        ++board->moveNumber;

    board->castling &= (board->pieces[KING] | board->pieces[ROOK]);
    board->key ^= otherSideKey;

    addMoveToHist(board);
}

void unmakeMove(Board* board, U16 move, Undo* undo) {
    getUndo(board, undo);

    if(MoveType(move) == CASTLE_MOVE) {
        int castlingRank = (!board->color == WHITE ? 0 : 7);
        U8 king = makePiece(KING, !board->color);

        if(board->squares[square(castlingRank, 6)] == king) {
            board->key ^= zobristCastlingKeys[!board->color * 2];
            movePiece(board, square(castlingRank, 5), square(castlingRank, 7));
        } else if(board->squares[square(castlingRank, 2)] == king) {
            board->key ^= zobristCastlingKeys[!board->color * 2 + 1];
            movePiece(board, square(castlingRank, 3), square(castlingRank, 0));
        }
    } else if(MoveType(move) == ENPASSANT_MOVE) {
        board->key ^= zobristEnpassantKeys[board->enpassantSquare];
        if(!board->color == WHITE)
            setPiece(board, PAWN, board->color, undo->enpassantSquare - 8);
        else
            setPiece(board, PAWN, board->color, undo->enpassantSquare + 8);
    }

    movePiece(board, MoveTo(move), MoveFrom(move));

     if(MoveType(move) == PROMOTION_MOVE) {
        setPiece(board, PAWN, !board->color, MoveFrom(move));
    }

    if(undo->capturedPiece)
        setPiece(board, pieceType(undo->capturedPiece), pieceColor(undo->capturedPiece), MoveTo(move));
    
    board->color = !board->color;
    if(board->color == BLACK)
        --board->moveNumber;
    
    board->key ^= otherSideKey;

    revertMoveFromHist(board);
}

void makeNullMove(Board* board) {
    board->color = !board->color;
    board->key ^= nullMoveKey;
}

void unmakeNullMove(Board* board) {
    board->color = !board->color;
    board->key ^= nullMoveKey;
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
    board->enpassantSquare = undo->enpassantSquare;
}

int attackedSquare(Board* board, int sq, int color) {
    U64 our = board->colours[color];
    U64 enemy = board->colours[!color];
    U64 occu = our | enemy;

    if(pawnAttacks[color][sq] & board->pieces[PAWN] & enemy)
        return 1;
    if(knightAttacks[sq] & enemy & board->pieces[KNIGHT])
        return 1;
    if(kingAttacks[sq] & board->pieces[KING] & enemy)
        return 1;

    U64 rookQueens = board->pieces[ROOK] | board->pieces[QUEEN];
    if(enemy & rookQueens & rookPossibleMoves[sq][getMagicIndex(occu & rookMagicMask[sq] & unSquareBitboard[sq], rookMagic[sq], rookPossibleMovesSize[sq])])
        return 1;

    U64 bishopQueens = board->pieces[BISHOP] | board->pieces[QUEEN];
    if(enemy & bishopQueens & bishopPossibleMoves[sq][getMagicIndex(occu & bishopMagicMask[sq] & unSquareBitboard[sq], bishopMagic[sq], bishopPossibleMovesSize[sq])])
        return 1;

    return 0;
}

int inCheck(Board* board, int color) {
    if(!(board->colours[color] & board->pieces[KING]))
        return 0;

    int kingPosition = firstOne(board->colours[color] & board->pieces[KING]);
    return attackedSquare(board, kingPosition, color);
}

void addMoveToHist(Board* board) {
    board->gameInfo->moveHistory[board->gameInfo->moveCount] = board->key;
    ++board->gameInfo->moveCount;
}

void revertMoveFromHist(Board* board) {
    --board->gameInfo->moveCount;
}

int isDraw(Board* board) {
    if(repeatCount(board) >= 2)
        return 1;
    if(board->ruleNumber >= 100)
        return 1;

    U64 pieces = board->colours[0] | board->colours[1];
    U64 piecesCount = popcount(pieces);

    if(piecesCount <= 2)
            return 1;

    if(piecesCount <= 3)
        return popcount(board->pieces[KNIGHT]) == 1 || popcount(board->pieces[BISHOP]) == 1;

    return (~board->pieces[BISHOP] & pieces) == board->pieces[KING];
}

int repeatCount(Board* board) {
    GameInfo* gameInfo = board->gameInfo;
    int rpt = 0;
    U64 currentKey = board->key;
    for(int i = gameInfo->moveCount - 1; i >= 0; --i) {
        rpt += gameInfo->moveHistory[i] == currentKey;
    }

    return rpt;
}

int havePromotionPawn(Board* board) {
    U64 ourPawns = board->pieces[PAWN] & board->colours[board->color];
    U64 occu = board->colours[board->color] | board->colours[!board->color];

    if(board->color == WHITE) {
        ourPawns &= ranks[6];
        return !!((ourPawns << 8) & ~occu);
    }

    ourPawns &= ranks[1];
    return !!((ourPawns >> 8) & ~occu);
}

int haveNoPawnMaterial(Board* board) {
    U64 occu = board->colours[WHITE] | board->colours[BLACK];
    return !!(~(board->pieces[PAWN] | board->pieces[KING]) & occu);
}

U64 attacksTo(Board* board, int sq) {
    U64 occu = board->colours[WHITE] | board->colours[BLACK];
    U64 knights, kings, rookQueens, bishopQueens;
    
    knights = board->pieces[KNIGHT];
    kings = board->pieces[KING];
    rookQueens = board->pieces[ROOK] | board->pieces[QUEEN];
    bishopQueens = board->pieces[BISHOP] | board->pieces[QUEEN];

    return (pawnAttacks[WHITE][sq] & board->pieces[PAWN] & board->colours[BLACK])
        | (pawnAttacks[BLACK][sq] & board->pieces[PAWN] & board->colours[WHITE])
        | (knightAttacks[sq] & knights)
        | (kingAttacks[sq] & kings)
        | (rookQueens & rookPossibleMoves[sq][getMagicIndex(occu & rookMagicMask[sq] & unSquareBitboard[sq], rookMagic[sq], rookPossibleMovesSize[sq])])
        | (bishopQueens & bishopPossibleMoves[sq][getMagicIndex(occu & bishopMagicMask[sq] & unSquareBitboard[sq], bishopMagic[sq], bishopPossibleMovesSize[sq])]);
}

int see(Board* board, int toSq, U8 taget, int fromSq, U8 aPiece) {
    int gain[32], d = 0, color = board->color;
    U64 mayXray = board->pieces[PAWN] | board->pieces[BISHOP] | board->pieces[ROOK] | board->pieces[QUEEN];
    U64 fromSet = (1ull << fromSq);
    U64 occu = board->colours[WHITE] | board->colours[BLACK];
    U64 attadef = attacksTo(board, toSq);
    gain[d] = pVal(board, pieceType(taget));

    do {
        d++;
        gain[d] = pVal(board, pieceType(aPiece)) - gain[d - 1];
        if(max(gain[d - 1], gain[d]) < 0)
            break;
        attadef ^= fromSet;
        occu ^= fromSet;
        if(fromSet & mayXray)
            attadef |= considerXrays(board, occu, attadef, toSq);
        color = !color;
        fromSet = getLeastValuablePiece(board, attadef, color, &aPiece);
    } while(fromSet);

    while (--d)
        gain[d-1]= -max(-gain[d-1], gain[d]);
    
    return gain[0];
}

U64 getLeastValuablePiece(Board* board, U64 attadef, int side, U8* piece) {
   U64 our = board->colours[side];
   for (*piece = makePiece(PAWN, side); *piece <= makePiece(KING, side); *piece += 2) {
      U64 subset = attadef & board->pieces[pieceType(*piece)] & our;
      if (subset)
         return  subset & -subset;
   }
   return 0;
}

U64 considerXrays(Board* board, U64 occu, U64 attackdef, int sq) {
    U64 rookQueens = board->pieces[ROOK] | board->pieces[QUEEN];
    U64 bishopQueens = board->pieces[BISHOP] | board->pieces[QUEEN];
    if(!(attackdef & minus1[sq]) && (occu & minus1[sq]))
        return squareBitboard[lastOne(occu & minus1[sq])] & rookQueens;
    else if(!(attackdef & minus7[sq]) && (occu & minus7[sq]))
        return squareBitboard[lastOne(occu & minus7[sq])] & bishopQueens;
    else if(!(attackdef & minus9[sq]) && (occu & minus9[sq]))
        return squareBitboard[lastOne(occu & minus9[sq])] & bishopQueens;
    else if(!(attackdef & minus8[sq]) && (occu & minus8[sq]))
        return squareBitboard[lastOne(occu & minus8[sq])] & rookQueens;
    else if(!(attackdef & plus1[sq]) && (occu & plus1[sq]))
        return squareBitboard[firstOne(occu & plus1[sq])] & rookQueens;
    else if(!(attackdef & plus7[sq]) && (occu & plus7[sq]))
        return squareBitboard[firstOne(occu & plus7[sq])] & bishopQueens;
    else if(!(attackdef & plus9[sq]) && (occu & plus9[sq]))
        return squareBitboard[firstOne(occu & plus9[sq])] & bishopQueens;
    else if(!(attackdef & plus8[sq]) && (occu & plus8[sq]))
        return squareBitboard[firstOne(occu & plus8[sq])] & rookQueens;

    return 0;
}