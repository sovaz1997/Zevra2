#include "movegen.h"

void movegen(Board* board, uint16_t* moveList) {
    int color = board->color;

    
    U64 our = board->colours[color]; //our pieces
    U64 enemy = board->colours[!color]; //enemy pieces
    U64 occu = (our | enemy); //all pieces
    U64 kingPos = board->pieces[KING] & enemy;

    //Rooks
    U64 mask = board->pieces[ROOK] & our;

    while(mask) {
        int from = firstOne(mask);
        U64 possibleMoves = ~kingPos & rookPossibleMoves[from][getMagicIndex(occu & rookMagicMask[from] & unSquareBitboard[from], rookMagic[from], rookPossibleMovesSize[from])];
        moveList = genMovesFromBitboard(from, possibleMoves & ~our, moveList);
        clearBit(&mask, from);
    }

    //Bishops
    mask = board->pieces[BISHOP] & our;

    while(mask) {
        int from = firstOne(mask);
        U64 possibleMoves = ~kingPos & bishopPossibleMoves[from][getMagicIndex(occu & bishopMagicMask[from] & unSquareBitboard[from], bishopMagic[from], bishopPossibleMovesSize[from])];
        moveList = genMovesFromBitboard(from, possibleMoves & ~our, moveList);
        clearBit(&mask, from);
    }

    //Queens
    mask = board->pieces[QUEEN] & our;

    while(mask) {
        int from = firstOne(mask);
        U64 possibleMoves = rookPossibleMoves[from][getMagicIndex(occu & rookMagicMask[from] & unSquareBitboard[from], rookMagic[from], rookPossibleMovesSize[from])];
        possibleMoves |= bishopPossibleMoves[from][getMagicIndex(occu & bishopMagicMask[from] & unSquareBitboard[from], bishopMagic[from], bishopPossibleMovesSize[from])];
        possibleMoves &= ~kingPos;
        moveList = genMovesFromBitboard(from, possibleMoves & ~our, moveList);
        clearBit(&mask, from);
    }

    //Knights
    mask = board->pieces[KNIGHT] & our;

    while(mask) {
        int from = firstOne(mask);
        U64 possibleMoves = ~kingPos & knightAttacks[from];
        moveList = genMovesFromBitboard(from, possibleMoves & ~our, moveList);
        clearBit(&mask, from);
    }

    //King
    mask = board->pieces[KING] & our;

    while(mask) {
        int from = firstOne(mask);

        U64 possibleMoves = ~kingPos & kingAttacks[from];
        moveList = genMovesFromBitboard(from, possibleMoves & ~our, moveList);
        clearBit(&mask, from);
    }

    //Pawns (moves)
    mask = board->pieces[PAWN] & our;

    while(mask) {
        int from = firstOne(mask);

        U64 possibleMoves = pawnMoves[color][from];

        if(color == WHITE) {
            if(possibleMoves & occu)
                possibleMoves &= ~plus8[firstOne(possibleMoves & occu) - 8];
        } else {
            if(possibleMoves & occu)
                possibleMoves &= ~minus8[lastOne(possibleMoves & occu) + 8];
        }

        moveList = genMovesFromBitboard(from, possibleMoves & ~(ranks[0] | ranks[7]), moveList);
        moveList = genPromoMovesFromBitboard(from, possibleMoves & (ranks[0] | ranks[7]), moveList);

        clearBit(&mask, from);
    }

    //Pawns (captures)
    mask = board->pieces[PAWN] & our;
    U64 rightAttacks, leftAttacks;

    if(color == WHITE) {
        rightAttacks = (mask << 9) & ~files[0] & enemy & ~kingPos;
        moveList = genPawnCaptures(rightAttacks, 9, moveList, NORMAL_MOVE);
        leftAttacks = (mask << 7) & ~files[7] & enemy & ~kingPos;
        moveList = genPawnCaptures(leftAttacks, 7, moveList, NORMAL_MOVE);

        if(board->enpassantSquare) {
            rightAttacks = (mask << 9) & ~files[0] & squareBitboard[board->enpassantSquare];
            moveList = genPawnCaptures(rightAttacks, 9, moveList, ENPASSANT_MOVE);
            leftAttacks = (mask << 7) & ~files[7] & squareBitboard[board->enpassantSquare];
            moveList = genPawnCaptures(leftAttacks, 7, moveList, ENPASSANT_MOVE);
        }
    } else {
        rightAttacks = (mask >> 9) & ~files[7] & enemy & ~kingPos;
        moveList = genPawnCaptures(rightAttacks, -9, moveList, NORMAL_MOVE);
        leftAttacks = (mask >> 7) & ~files[0] & enemy & ~kingPos;
        moveList = genPawnCaptures(leftAttacks, -7, moveList, NORMAL_MOVE);

        if(board->enpassantSquare) {
            rightAttacks = (mask >> 9) & ~files[7] & squareBitboard[board->enpassantSquare];
            moveList = genPawnCaptures(rightAttacks, -9, moveList, ENPASSANT_MOVE);
            leftAttacks = (mask >> 7) & ~files[0] & squareBitboard[board->enpassantSquare];
            moveList = genPawnCaptures(leftAttacks, -7, moveList, ENPASSANT_MOVE);
        }
    }

    //Castlings
    U8 king = makePiece(KING, color);
    U8 rook = makePiece(ROOK, color);
    int castlingRank = (color == WHITE ? 0 : 7);

    if((board->castling & shortCastlingBitboard[color]) == shortCastlingBitboard[color]) {
        if(board->squares[square(castlingRank, 4)] == king
        && board->squares[square(castlingRank, 7)] == rook
        && !board->squares[square(castlingRank, 5)]
        && !board->squares[square(castlingRank, 6)]) {
            if(!attackedSquare(board, square(castlingRank, 4), color)
            && !attackedSquare(board, square(castlingRank, 5), color)
            && !attackedSquare(board, square(castlingRank, 6), color)) {
                *(moveList++) = MakeMove(square(castlingRank, 4), square(castlingRank, 6), CASTLE_MOVE);
            }
        }
    }

    if((board->castling & longCastlingBitboard[color]) == longCastlingBitboard[color]) {
        if(board->squares[square(castlingRank, 4)] == king
        && board->squares[square(castlingRank, 0)] == rook
        && !board->squares[square(castlingRank, 3)]
        && !board->squares[square(castlingRank, 2)]
        && !board->squares[square(castlingRank, 1)]
        ) {
            if(!attackedSquare(board, square(castlingRank, 4), color)
            && !attackedSquare(board, square(castlingRank, 3), color)
            && !attackedSquare(board, square(castlingRank, 2), color)) {
                *(moveList++) = MakeMove(square(castlingRank, 4), square(castlingRank, 2), CASTLE_MOVE);
            }
        }
    }

    *moveList = 0;
}

void attackgen(Board* board, uint16_t* moveList) {
    int color = board->color;
    
    U64 our = board->colours[color]; //our pieces
    U64 enemy = board->colours[!color]; //enemy pieces
    U64 occu = (our | enemy); //all pieces
    U64 kingPos = board->pieces[KING] & enemy;

    //Rooks
    U64 mask = board->pieces[ROOK] & our;

    while(mask) {

        int from = firstOne(mask);
        U64 possibleMoves = ~kingPos & rookPossibleMoves[from][getMagicIndex(occu & rookMagicMask[from] & unSquareBitboard[from], rookMagic[from], rookPossibleMovesSize[from])];
        moveList = genMovesFromBitboard(from, possibleMoves & ~our & enemy, moveList);
        clearBit(&mask, from);
    }

    //Bishops
    mask = board->pieces[BISHOP] & our;

    while(mask) {
        int from = firstOne(mask);
        U64 possibleMoves = ~kingPos & bishopPossibleMoves[from][getMagicIndex(occu & bishopMagicMask[from] & unSquareBitboard[from], bishopMagic[from], bishopPossibleMovesSize[from])];
        moveList = genMovesFromBitboard(from, possibleMoves & ~our & enemy, moveList);
        clearBit(&mask, from);
    }

    //Queens
    mask = board->pieces[QUEEN] & our;

    while(mask) {
        int from = firstOne(mask);
        U64 possibleMoves = rookPossibleMoves[from][getMagicIndex(occu & rookMagicMask[from] & unSquareBitboard[from], rookMagic[from], rookPossibleMovesSize[from])];
        possibleMoves |= bishopPossibleMoves[from][getMagicIndex(occu & bishopMagicMask[from] & unSquareBitboard[from], bishopMagic[from], bishopPossibleMovesSize[from])];
        possibleMoves &= ~kingPos;
        moveList = genMovesFromBitboard(from, possibleMoves & ~our & enemy, moveList);
        clearBit(&mask, from);
    }

    //Knights
    mask = board->pieces[KNIGHT] & our;

    while(mask) {
        int from = firstOne(mask);

        U64 possibleMoves = ~kingPos & knightAttacks[from];
        moveList = genMovesFromBitboard(from, possibleMoves & ~our & enemy, moveList);
        clearBit(&mask, from);
    }

    //King
    mask = board->pieces[KING] & our;

    while(mask) {
        int from = firstOne(mask);

        U64 possibleMoves = ~kingPos & kingAttacks[from];
        moveList = genMovesFromBitboard(from, possibleMoves & ~our & enemy, moveList);
        clearBit(&mask, from);
    }

    //Pawn (captures)
    mask = board->pieces[PAWN] & our;
    U64 rightAttacks, leftAttacks;

    if(color == WHITE) {
        rightAttacks = (mask << 9) & ~files[0] & enemy & ~kingPos;
        moveList = genPawnCaptures(rightAttacks, 9, moveList, NORMAL_MOVE);
        leftAttacks = (mask << 7) & ~files[7] & enemy & ~kingPos;
        moveList = genPawnCaptures(leftAttacks, 7, moveList, NORMAL_MOVE);

        if(board->enpassantSquare) {
            rightAttacks = (mask << 9) & ~files[0] & squareBitboard[board->enpassantSquare];
            moveList = genPawnCaptures(rightAttacks, 9, moveList, ENPASSANT_MOVE);
            leftAttacks = (mask << 7) & ~files[7] & squareBitboard[board->enpassantSquare];
            moveList = genPawnCaptures(leftAttacks, 7, moveList, ENPASSANT_MOVE);
        }
    } else {
        rightAttacks = (mask >> 9) & ~files[7] & enemy & ~kingPos;
        moveList = genPawnCaptures(rightAttacks, -9, moveList, NORMAL_MOVE);
        leftAttacks = (mask >> 7) & ~files[0] & enemy & ~kingPos;
        moveList = genPawnCaptures(leftAttacks, -7, moveList, NORMAL_MOVE);

        if(board->enpassantSquare) {
            rightAttacks = (mask >> 9) & ~files[7] & squareBitboard[board->enpassantSquare];
            moveList = genPawnCaptures(rightAttacks, -9, moveList, ENPASSANT_MOVE);
            leftAttacks = (mask >> 7) & ~files[0] & squareBitboard[board->enpassantSquare];
            moveList = genPawnCaptures(leftAttacks, -7, moveList, ENPASSANT_MOVE);
        }
    }

    *moveList = 0;
}

uint16_t* genMovesFromBitboard(int from, U64 bitboard, uint16_t* moveList) {
    while(bitboard) {
        int to = firstOne(bitboard);
        *(moveList++) = MakeMove(from, to, NORMAL_MOVE);
        clearBit(&bitboard, to);
    }
    return moveList;
}

uint16_t* genPromoMovesFromBitboard(int from, U64 bitboard, uint16_t* moveList) {
    while(bitboard) {
        int to = firstOne(bitboard);
        *(moveList++) = MakeMove(from, to, QUEEN_PROMOTE_MOVE);
        *(moveList++) = MakeMove(from, to, ROOK_PROMOTE_MOVE);
        *(moveList++) = MakeMove(from, to, KNIGHT_PROMOTE_MOVE);
        *(moveList++) = MakeMove(from, to, BISHOP_PROMOTE_MOVE);
        clearBit(&bitboard, to);
    }
    return moveList;
}

uint16_t* genPawnCaptures(U64 to, int diff, uint16_t* moveList, U16 flags) {
    
    while(to) {
        int toSq = firstOne(to);
        
        if(rankOf(toSq) == 0 || rankOf(toSq) == 7) {
            *(moveList++) = MakeMove(toSq - diff, toSq, QUEEN_PROMOTE_MOVE);
            *(moveList++) = MakeMove(toSq - diff, toSq, ROOK_PROMOTE_MOVE);
            *(moveList++) = MakeMove(toSq - diff, toSq, KNIGHT_PROMOTE_MOVE);
            *(moveList++) = MakeMove(toSq - diff, toSq, BISHOP_PROMOTE_MOVE);
        } else {
            *(moveList++) = MakeMove(toSq - diff, toSq, NORMAL_MOVE | flags);
        }
            
        
        clearBit(&to, toSq);
    }
    return moveList;
}