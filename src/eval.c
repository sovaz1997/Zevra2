#include "eval.h"
#include "score.h"
#include <math.h>

//Material
int PAWN_EV = 100;
int KNIGHT_EV = 322;
int BISHOP_EV = 322;
int ROOK_EV = 500;
int QUEEN_EV = 923;

int PAWN_EV_EG = 100;
int KNIGHT_EV_EG = 322;
int BISHOP_EV_EG = 322;
int ROOK_EV_EG = 500;
int QUEEN_EV_EG = 923;

//Mobility bonuses
int QueenMobility[28] = { -30, -20, -10, -52, -47, 34, -3, 18, 23, 24, 18, 29, 36, 39, 42, 47, 50, 42, 48, 49, 52, 46, 55, 52, 37, 76, 45, 42, };
int QueenMobilityEG[28] = { -30, -20, -10, -52, -47, 34, -3, 18, 23, 24, 18, 29, 36, 39, 42, 47, 50, 42, 48, 49, 52, 46, 55, 52, 37, 76, 45, 42, };

int RookMobility[15] = {-402, -51, -57, -22, 0, 0, 8, 8, 13, 21, 25, 29, 35, 34, 35, };
int RookMobilityEG[15] = {-402, -51, -57, -22, 0, 0, 8, 8, 13, 21, 25, 29, 35, 34, 35, };

int BishopMobility[14] = {-13, -39, -23, -9, 2, 6, 15, 19, 21, 25, 27, 39, 21, 38,};
int BishopMobilityEG[14] = {-13, -39, -23, -9, 2, 6, 15, 19, 21, 25, 27, 39, 21, 38,};

int KnightMobility[8] = {-101, -46, -24, -18, -19, -22, -21, -14, };
int KnightMobilityEG[8] = {-101, -46, -24, -18, -19, -22, -21, -14, };

//additional bonuses and penalties
int PassedPawnBonus[8] = {0, 0, -5, 5, 36, 92, 163, 0, };
int PassedPawnBonusEG[8] = {0, 0, -5, 5, 36, 92, 163, 0, };

int DoublePawnsPenalty = -30;
int DoublePawnsPenaltyEG = -30;

int IsolatedPawnPenalty = -4;
int IsolatedPawnPenaltyEG = -4;

int RookOnOpenFileBonus = 20;
int RookOnOpenFileBonusEG = 20;

int RookOnPartOpenFileBonus = 24;
int RookOnPartOpenFileBonusEG = 24;

int KingDangerFactor = 602;

int DoubleBishopsBonusMG = 38;
int DoubleBishopsBonusEG = 30;

int DoubleBishopsBonus() {
    return getScore2(DoubleBishopsBonusMG, DoubleBishopsBonusEG, stage);
}

int fullEval(Board *board) {

    int eval = 0;
    stage = stageGame(board);

    //Base eval (Material + PSQT)
    eval += materialEval(board);
    eval += psqtEval(board);

    //Mobility eval
    eval += (mobilityAndKingDangerEval(board, WHITE) - mobilityAndKingDangerEval(board, BLACK));

    //Pieces eval
    eval += (pawnsEval(board, WHITE) - pawnsEval(board, BLACK));
    eval += bishopsEval(board);
    eval += (rooksEval(board, WHITE) - rooksEval(board, BLACK));

    //King safety
    eval += (kingEval(board, WHITE) - kingEval(board, BLACK));

    int normalizedEval = round((double)eval / (double)PAWN_EV * 100.);

    return (board->color == WHITE ? normalizedEval : -normalizedEval);
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
    eval += (psqtPieceEval(board, mask, kingPST) * stage / 98. + psqtPieceEval(board, mask, egKingPST) * (98. - stage) / 98.);

    return eval;
}

int psqtPieceEval(Board *board, U64 mask, const int *pstTable) {
    int eval = 0;

    while (mask) {
        int sq = firstOne(mask);
        if (squareBitboard[sq] & board->colours[WHITE])
            eval += *(pstTable + square(7 - rankOf(sq), fileOf(sq)));
        else
            eval -= *(pstTable + sq);

        clearBit(&mask, sq);
    }

    return eval;
}

int kingDanger(int attacksCount) {
    double normalized = (attacksCount / 100. * 10.) - 5;

    return KingDangerFactor * (1. / (1. + exp(-normalized))) - 4;
}

int mobilityAndKingDangerEval(Board *board, int color) {
    int eval = 0;

    U64 our = board->colours[color]; //our pieces
    U64 enemy = board->colours[!color]; //enemy pieces
    U64 occu = (our | enemy); //all pieces

    U64 mask = board->pieces[PAWN] & enemy;
    U64 pAttacks;

    if (!color == WHITE)
        pAttacks = ((mask << 9) & ~files[0]) | ((mask << 7) & ~files[7]);
    else
        pAttacks = ((mask >> 9) & ~files[7]) | ((mask >> 7) & ~files[0]);

    U64 possibleSq = ~pAttacks;


    int enemyKingPos = firstOne(enemy & board->pieces[KING]);
    U64 enemyKingDangerCells = kingAttacks[enemyKingPos] & ~enemy;

    int kingDangerValue = 0;

    //Rooks mobility
    mask = board->pieces[ROOK] & our;

    while (mask) {
        int from = firstOne(mask);
        U64 possibleMoves = rookPossibleMoves[from][getMagicIndex(occu & rookMagicMask[from] & unSquareBitboard[from],
                                                                  rookMagic[from], rookPossibleMovesSize[from])];
        eval += RookMobility[popcount(possibleMoves & possibleSq)];

        kingDangerValue += 3 * popcount(possibleMoves & enemyKingDangerCells);

        clearBit(&mask, from);
    }

    //Bishops mobility
    mask = board->pieces[BISHOP] & our;

    while (mask) {
        int from = firstOne(mask);
        U64 possibleMoves = bishopPossibleMoves[from][getMagicIndex(
                occu & bishopMagicMask[from] & unSquareBitboard[from], bishopMagic[from],
                bishopPossibleMovesSize[from])];
        eval += BishopMobility[popcount(possibleMoves & possibleSq)];

        kingDangerValue += 2 * popcount(possibleMoves & enemyKingDangerCells);

        clearBit(&mask, from);
    }

    //Queens mobility
    mask = board->pieces[QUEEN] & our;

    while (mask) {
        int from = firstOne(mask);
        U64 possibleMoves = (
                (rookPossibleMoves[from][getMagicIndex(occu & rookMagicMask[from] & unSquareBitboard[from],
                                                       rookMagic[from], rookPossibleMovesSize[from])])
                | (bishopPossibleMoves[from][getMagicIndex(occu & bishopMagicMask[from] & unSquareBitboard[from],
                                                           bishopMagic[from], bishopPossibleMovesSize[from])])
        );
        eval += QueenMobility[popcount(possibleMoves & possibleSq)];

        kingDangerValue += 5 * popcount(possibleMoves & enemyKingDangerCells);

        clearBit(&mask, from);
    }

    //Knights mobility
    mask = board->pieces[KNIGHT] & our;

    while (mask) {
        int from = firstOne(mask);
        U64 possibleMoves = knightAttacks[from];
        eval += KnightMobility[popcount(possibleMoves & possibleSq)];

        kingDangerValue += 2 * popcount(possibleMoves & enemyKingDangerCells);

        clearBit(&mask, from);
    }

    return eval + KingDanger[kingDangerValue];
}

int pawnsEval(Board *board, int color) {
    int eval = 0;

    U64 ourPawns = board->colours[color] & board->pieces[PAWN];
    U64 enemyPawns = board->colours[!color] & board->pieces[PAWN];

    //isolated pawn bonus
    eval += (IsolatedPawnsHash[horizontalScan(ourPawns)]);

    //double pawns bonus
    for (int f = 0; f < 8; ++f)
        eval += DoublePawnsPenalty * (popcount(ourPawns & files[f]) > 1);

    //passed pawn bonus
    while (ourPawns) {
        int sq = firstOne(ourPawns);
        if (color == WHITE) {
            if (!(plus8[sq] & enemyPawns))
                eval += getPassedPawnBonus(sq, color);
        } else {
            if (!(minus8[sq] & enemyPawns))
                eval += getPassedPawnBonus(sq, color);
        }
        clearBit(&ourPawns, sq);
    }

    return eval;
}

int bishopsEval(Board *board) {
    int eval = 0;
    int score = DoubleBishopsBonus();

    //double bishops bonus
    eval += (score * (popcount(board->pieces[BISHOP] & board->colours[WHITE]) > 1) -
             score * (popcount(board->pieces[BISHOP] & board->colours[BLACK]) > 1));

    return eval;
}

int getPassedPawnBonus(int sq, int color) {
    if (color == WHITE)
        return -pawnPST[square(7 - rankOf(sq), fileOf(sq))] + PassedPawnBonus[rankOf(sq)];

    return -pawnPST[square(rankOf(sq), fileOf(sq))] + PassedPawnBonus[7 - rankOf(sq)];
}

int kingEval(Board *board, int color) {
    int eval = 0;
    U64 enemy = board->colours[!color];
    int kingPos = firstOne(board->pieces[KING] & board->colours[color]);

    while (enemy) {
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

void initEval() {
    initDependencyEval();

    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j)
            distanceBonus[i][j] = 14 - (abs(rankOf(i) - rankOf(j)) + abs(fileOf(i) - fileOf(j)));
    }
}

void initDependencyEval() {
    for (int i = 0; i < 100; i++) {
        KingDanger[i] = kingDanger(i);
    }

    //Isolated pawn hash init
    for (int i = 0; i < 256; ++i) {
        IsolatedPawnsHash[i] = 0;
        for (int f = 0; f < 8; ++f) {
            int leftEmpty = 1, rightEmpty = 1;

            if (f < 7)
                rightEmpty = !getBit8(i, f + 1);
            if (f > 0)
                leftEmpty = !getBit8(i, f - 1);

            IsolatedPawnsHash[i] += IsolatedPawnPenalty * (leftEmpty && rightEmpty && getBit(i, f));
        }
    }
}

int pVal(int n) {
    switch (n) {
        case 0:
            return 0;
        case 1:
            return PAWN_EV;
        case 2:
            return KNIGHT_EV;
        case 3:
            return BISHOP_EV;
        case 4:
            return ROOK_EV;
        case 5:
            return QUEEN_EV;
        case 6:
            return 0;
    }
}

int stageGame(Board *board) {
    return popcount(board->pieces[QUEEN]) * 12 + popcount(board->pieces[ROOK]) * 8 +
           popcount(board->pieces[BISHOP]) * 5 + popcount(board->pieces[KNIGHT]) * 5;
}

int rooksEval(Board *board, int color) {
    int eval = 0;
    U64 our = board->colours[color];

    U64 rookMask = board->pieces[ROOK] & our;

    //rook on open file bonus

    while (rookMask) {
        int sq = firstOne(rookMask);

        U64 file = files[fileOf(sq)];

        int ourPawnsOnFile = popcount(file & board->pieces[PAWN] & board->colours[color]) > 0;
        int enemyPawnsOnFile = popcount(file & board->pieces[PAWN] & board->colours[!color]) > 0;

        if (!ourPawnsOnFile) {
            if (enemyPawnsOnFile) {
                eval += RookOnPartOpenFileBonus;
            } else {
                eval += RookOnOpenFileBonus;
            }
        }

        clearBit(&rookMask, sq);
    }

    return eval;
}

U8 horizontalScan(U64 bitboard) {
    return (!!(bitboard & files[0])) | (!!(bitboard & files[1]) << 1)
           | (!!(bitboard & files[2])) << 2 | (!!(bitboard & files[3]) << 3)
           | (!!(bitboard & files[4])) << 4 | (!!(bitboard & files[5]) << 5)
           | (!!(bitboard & files[6])) << 6 | (!!(bitboard & files[7]) << 7);
}