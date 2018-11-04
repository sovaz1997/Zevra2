#include "eval.h"

int fullEval(Board* board) {
    int eval = 0;
    stage = stageGame(board);

    //Базовая оценка
    eval += materialEval(board);
    eval += psqtEval(board);

    //Оценка мобильности
    eval += (mobilityEval(board, WHITE) - mobilityEval(board, BLACK));

    //Разделенная оценка фигур
    eval += (pawnsEval(board, WHITE) - pawnsEval(board, BLACK));
    eval += bishopsEval(board);
    eval += (rooksEval(board, WHITE) - rooksEval(board, BLACK));

    //Безопасность короля
    eval += (kingEval(board, WHITE) - kingEval(board, BLACK));

    return (board->color == WHITE ? eval : -eval);
}

int materialEval(Board* board) {
    int eval = 0;
    
    int wpCount = popcount(board->pieces[PAWN] & board->colours[WHITE]);
    int bpCount = popcount(board->pieces[PAWN] & board->colours[BLACK]);
    int wnCount = popcount(board->pieces[KNIGHT] & board->colours[WHITE]);
    int bnCount = popcount(board->pieces[KNIGHT] & board->colours[BLACK]);
    int wbCount = popcount(board->pieces[BISHOP] & board->colours[WHITE]);
    int bbCount = popcount(board->pieces[BISHOP] & board->colours[BLACK]);
    int wrCount = popcount(board->pieces[ROOK] & board->colours[WHITE]);
    int brCount = popcount(board->pieces[ROOK] & board->colours[BLACK]);
    int wqCount = popcount(board->pieces[QUEEN] & board->colours[WHITE]);
    int bqCount = popcount(board->pieces[QUEEN] & board->colours[BLACK]);

    eval += getScore(PAWN_EV, stage) * (wpCount - bpCount);
    eval += KNIGHT_EV * (wnCount - bnCount);
    eval += BISHOP_EV * (wbCount - bbCount);
    eval += ROOK_EV * (wrCount - brCount);
    eval += QUEEN_EV * (wqCount - bqCount);

    return eval;
}

int psqtEval(Board* board) {
    int eval = 0;

    U64 mask = board->pieces[PAWN];
    eval += psqtPieceEval(board, mask, pawnPST);
    
    mask = board->pieces[KNIGHT];
    eval += psqtPieceEval(board, mask, knightPST);

    mask = board->pieces[BISHOP];
    eval += psqtPieceEval(board, mask, bishopPST);

    mask = board->pieces[ROOK];
    eval += psqtPieceEval(board, mask, rookPST);

    mask = board->pieces[QUEEN];
    eval += psqtPieceEval(board, mask, queenPST);

    mask = board->pieces[KING];
    eval += (psqtPieceEval(board, mask, kingPST) * stage / 98. + psqtPieceEval(board, mask, egKingPST) * (98. - stage) / 98.);
    
    return eval;
}

int psqtPieceEval(Board* board, U64 mask, const int* pstTable) {
    int eval = 0;

    while(mask) {
        int sq = firstOne(mask);
        if(squareBitboard[sq] & board->colours[WHITE]) {
            eval += *(pstTable + square(7 - rankOf(sq), fileOf(sq)));
            
        } else {
            eval -= *(pstTable + sq);
        }
        clearBit(&mask, sq);
    }

    return eval;
}

int mobilityEval(Board* board, int color) {
    int eval = 0;

    U64 our = board->colours[color]; //наши фигуры
    U64 enemy = board->colours[!color]; //чужие фигуры
    U64 occu = (our | enemy); //все фигуры

    U64 mask = board->pieces[PAWN] & enemy;
    U64 pawnAttacks;
    if(!color == WHITE) {
        pawnAttacks = ((mask << 9) & ~files[0]) | ((mask << 7) & ~files[7]);
    } else {
        pawnAttacks = ((mask >> 9) & ~files[7]) | ((mask >> 7) & ~files[0]);
    }
    U64 possibleSq = ~pawnAttacks;

    //Ладья
    mask = board->pieces[ROOK] & our;

    while(mask) {
        int from = firstOne(mask);
        U64 possibleMoves = rookPossibleMoves[from][getMagicIndex(occu & rookMagicMask[from] & unSquareBitboard[from], rookMagic[from], rookPossibleMovesSize[from])];
        eval += RookMobility[popcount(possibleMoves & possibleSq)];
        clearBit(&mask, from);
    }

    //Слон
    mask = board->pieces[BISHOP] & our;

    while(mask) {
        int from = firstOne(mask);
        U64 possibleMoves = bishopPossibleMoves[from][getMagicIndex(occu & bishopMagicMask[from] & unSquareBitboard[from], bishopMagic[from], bishopPossibleMovesSize[from])];
        eval += BishopMobility[popcount(possibleMoves & possibleSq)];
        clearBit(&mask, from);
    }

    //Ферзь
    mask = board->pieces[QUEEN] & our;

    while(mask) {
        int from = firstOne(mask);
        U64 possibleMoves = rookPossibleMoves[from][getMagicIndex(occu & rookMagicMask[from] & unSquareBitboard[from], rookMagic[from], rookPossibleMovesSize[from])];
        eval += QueenMobility[popcount(possibleMoves & possibleSq)];
        clearBit(&mask, from);
    }

    //Конь
    mask = board->pieces[KNIGHT] & our;

    while(mask) {
        int from = firstOne(mask);
        U64 possibleMoves = knightAttacks[from];
        eval += KnightMobility[popcount(possibleMoves & possibleSq)];
        clearBit(&mask, from);
    }

    return eval;
}

int pawnsEval(Board* board, int color) {
    int eval = 0;
    
    U64 ourPawns = board->colours[color] & board->pieces[PAWN];
    U64 enemyPawns = board->colours[!color] & board->pieces[PAWN];

    while(ourPawns) {
        int sq = firstOne(ourPawns);
        if(color == WHITE) {
            if(!(plus8[sq] & enemyPawns)) {
                eval += getPassedPawnBonus(sq, color);
            }
        } else {
            if(!(minus8[sq] & enemyPawns)) {
                eval += getPassedPawnBonus(sq, color);
            }
        }
        clearBit(&ourPawns, sq);
    }

    for(int f = 0; f < 8; ++f) {
        eval -= DoublePawnsPenalty * (popcount(ourPawns & files[f]) > 1);
    }

    return eval;
}

int bishopsEval(Board* board) {
    int eval = 0;
    int score = getScore(DoubleBishopsBonus, stage);

    //Бонус за наличие 2-х слонов
    eval += (score * (popcount(board->pieces[BISHOP] & board->colours[WHITE]) > 1) -
        score * (popcount(board->pieces[BISHOP] & board->colours[BLACK]) > 1));

    return eval;
}

int getPassedPawnBonus(int sq, int color) {
    if(color == WHITE) {
        return -pawnPST[square(7 - rankOf(sq), fileOf(sq))] + PassedPawnBonus[rankOf(sq)];
    }
    
    return -pawnPST[square(rankOf(sq), fileOf(sq))] + PassedPawnBonus[7 - rankOf(sq)];
}

int kingSafety(Board* board, int color) {
    int attacks = 0;
    int kingPos = firstOne(board->pieces[KING] & board->colours[color]);
    U64 our = board->colours[color];
    U64 mask = (kingAttacks[kingPos] & ~our | squareBitboard[kingPos]);

    while(mask) {
        int sq = firstOne(mask);
        attacks += attackCount(board, sq, color);
        clearBit(&mask, sq);
    }

    return -SafetyTable[min(99, attacks)];
}

int attackCount(Board* board, int sq, int color) {
    int attacks = 0;

    U64 our = board->colours[color];
    U64 enemy = board->colours[!color];
    U64 occu = our | enemy;

    U64 possibleAttackers = attacksTo(board, sq);

    attacks += 5 * popcount(possibleAttackers & enemy & board->pieces[QUEEN]);
    attacks += 3 * popcount(possibleAttackers & enemy & board->pieces[ROOK]);
    attacks += 2 * popcount(possibleAttackers & enemy & board->pieces[KNIGHT]);
    attacks += 2 * popcount(possibleAttackers & enemy & board->pieces[BISHOP]);
    return attacks;
}

int kingEval(Board* board, int color) {
    int eval = 0;
    U64 enemy = board->colours[!color];
    int kingPos = firstOne(board->pieces[KING] & board->colours[color]);

    while(enemy) {
        int sq = firstOne(enemy);
        eval -= 4 * distanceBonus[sq][kingPos] * (pieceType(board->squares[sq]) == QUEEN);
        eval -= 3 * distanceBonus[sq][kingPos] * (pieceType(board->squares[sq]) == ROOK);
        eval -= 2 * distanceBonus[sq][kingPos] * (pieceType(board->squares[sq]) == KNIGHT);
        eval -= 2 * distanceBonus[sq][kingPos] * (pieceType(board->squares[sq]) == BISHOP);
        clearBit(&enemy, sq);
    }
    return eval;
}

int mateScore(int eval) {
    return (eval >= MATE_SCORE - 100 || eval <= -MATE_SCORE + 100);
}

int closeToMateScore(int eval) {
    return (eval >= MATE_SCORE / 2 || eval <= -MATE_SCORE / 2);
}

void initEval() {
    for(int i = 0; i < 64; ++i) {
        for(int j = 0; j < 64; ++j) {
            distanceBonus[i][j] = 14 - (abs(rankOf(i) - rankOf(j)) + abs(fileOf(i) - fileOf(j)));
        }   
    }
}

int stageGame(Board* board) {
    return popcount(board->pieces[QUEEN]) * 12 + popcount(board->pieces[ROOK]) * 8 + popcount(board->pieces[BISHOP]) * 5 + popcount(board->pieces[KNIGHT]) * 5;
}

int rooksEval(Board* board, int color) {
    int eval = 0;
    U64 our = board->colours[color];
    U64 enemy = board->colours[!color];
    U64 occu = our | enemy;
    
    U64 rookMask = board->pieces[ROOK] & our;
    
    int bonus = getScore(RookOnOpenFileBonus, stage);

    while(rookMask) {
        int sq = firstOne(rookMask);
        eval += bonus * !((color == WHITE ? plus8[sq] : minus8[sq]) & board->pieces[PAWN] & files[fileOf(sq)]);
        clearBit(&rookMask, sq);
    }

    return eval;
}