#include "movegen.h"

void movegen(Board* board, uint16_t* moveList) {
    int color = board->color;

    
    U64 our = board->colours[color]; //наши фигуры
    U64 enemy = board->colours[!color]; //чужие фигуры
    U64 occu = (our | enemy); //все фигуры

    //Ладья
    U64 mask = board->pieces[ROOK] & our;

    while(mask) {

        int from = firstOne(mask);
        U64 possibleMoves = rookPossibleMoves[from][getMagicIndex(occu & rookMagicMask[from] & unSquareBitboard[from], rookMagic[from], rookPossibleMovesSize[from])];
        //printBitboard(occu & rookMagicMask[from] & ~bitboardCell(from));
        //printf("%d\n", getMagicIndex(occu & rookMagicMask[from] & ~bitboardCell(from), rookMagic[from], rookPossibleMovesSize[from]));

        moveList = genMovesFromBitboard(from, possibleMoves & ~our, moveList);
        clearBit(&mask, from);
    }

    //Слон
    mask = board->pieces[BISHOP] & our;

    while(mask) {
        int from = firstOne(mask);
        U64 possibleMoves = bishopPossibleMoves[from][getMagicIndex(occu & bishopMagicMask[from] & unSquareBitboard[from], bishopMagic[from], bishopPossibleMovesSize[from])];
        moveList = genMovesFromBitboard(from, possibleMoves & ~our, moveList);
        clearBit(&mask, from);
    }

    //Ферзь
    mask = board->pieces[QUEEN] & our;

    while(mask) {
        int from = firstOne(mask);
        U64 possibleMoves = rookPossibleMoves[from][getMagicIndex(occu & rookMagicMask[from] & unSquareBitboard[from], rookMagic[from], rookPossibleMovesSize[from])];
        possibleMoves |= bishopPossibleMoves[from][getMagicIndex(occu & bishopMagicMask[from] & unSquareBitboard[from], bishopMagic[from], bishopPossibleMovesSize[from])];
        moveList = genMovesFromBitboard(from, possibleMoves & ~our, moveList);
        clearBit(&mask, from);
    }

    //Конь
    mask = board->pieces[KNIGHT] & our;

    while(mask) {
        int from = firstOne(mask);

        U64 possibleMoves = knightAttacks[from];
        moveList = genMovesFromBitboard(from, possibleMoves & ~our, moveList);
        clearBit(&mask, from);
    }

    //Король
    mask = board->pieces[KING] & our;

    while(mask) {
        int from = firstOne(mask);

        U64 possibleMoves = kingAttacks[from];
        moveList = genMovesFromBitboard(from, possibleMoves & ~our, moveList);
        clearBit(&mask, from);
    }

    //Пешка (ходы вперед)
    mask = board->pieces[PAWN] & our;

    while(mask) {
        int from = firstOne(mask);

        U64 possibleMoves = pawnMoves[color][from];

        if(color == WHITE) {
            if(possibleMoves & occu) {
                possibleMoves &= ~plus8[firstOne(possibleMoves & occu) - 8];
            }
        } else {
            if(possibleMoves & occu) {
                possibleMoves &= ~minus8[lastOne(possibleMoves & occu) + 8];
            }
        }


        moveList = genMovesFromBitboard(from, possibleMoves & ~(ranks[0] | ranks[7]), moveList);
        moveList = genPromoMovesFromBitboard(from, possibleMoves & (ranks[0] | ranks[7]), moveList);

        clearBit(&mask, from);
    }

    //Пешка (взятия)
    mask = board->pieces[PAWN] & our;

    if(color == WHITE) {
        U64 rightAttacks = (mask << 9) & ~files[0] & enemy;
        moveList = genPawnCaptures(rightAttacks, 9, moveList);
        U64 leftAttacks = (mask << 7) & ~files[7] & enemy;
        moveList = genPawnCaptures(leftAttacks, 7, moveList);
    } else {
        U64 rightAttacks = (mask >> 9) & ~files[7] & enemy;
        moveList = genPawnCaptures(rightAttacks, -9, moveList);

        U64 leftAttacks = (mask >> 7) & ~files[0] & enemy;
        moveList = genPawnCaptures(leftAttacks, -7, moveList);
    }

    //Рокировки
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

uint16_t* genPawnCaptures(U64 to, int diff, uint16_t* moveList) {
    
    while(to) {
        int toSq = firstOne(to);
        
        if(rankOf(toSq) == 0 || rankOf(toSq) == 7) {
            *(moveList++) = MakeMove(toSq - diff, toSq, QUEEN_PROMOTE_MOVE);
            *(moveList++) = MakeMove(toSq - diff, toSq, ROOK_PROMOTE_MOVE);
            *(moveList++) = MakeMove(toSq - diff, toSq, KNIGHT_PROMOTE_MOVE);
            *(moveList++) = MakeMove(toSq - diff, toSq, BISHOP_PROMOTE_MOVE);
        } else {
            *(moveList++) = MakeMove(toSq - diff, toSq, NORMAL_MOVE);
        }
            
        
        clearBit(&to, toSq);
    }
    return moveList;
}