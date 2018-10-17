#include "eval.h"

int QueenMobility[28] = {
    -30, -20, -10, 0, 5, 10, 12, 15, 18, 20, 25, 30, 32, 35,
    40, 45, 50, 55, 57, 60, 63, 65, 70, 75, 80, 85, 90, 95
};
int RookMobility[15] = {-30, -20, -10, 0, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80};
int BishopMobility[14] = {-30, -10, 5, 15, 20, 25, 35, 40, 45, 50, 55, 60, 65, 70};
int KnightMobility[14] = {-50, -25, -10, -2, 5, 10, 15, 25};

int fullEval(Board* board) {
    int eval = 0;

    //Базовая оценка
    eval += materialEval(board);
    eval += psqtEval(board);

    //Оценка мобильности
    eval += mobilityEval(board, WHITE);
    eval -= mobilityEval(board, BLACK);

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

    eval += PAWN_EV * (wpCount - bpCount);
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
    eval += psqtPieceEval(board, mask, kingPST);
    
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