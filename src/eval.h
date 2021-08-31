#ifndef EVAL_H
#define EVAL_H

#include "board.h"
#include "psqt.h"
#include "score.h"

//Piece weights
enum {
    PAWN_EV = 100,
    KNIGHT_EV = 300,
    BISHOP_EV = 330,
    ROOK_EV = 550,
    QUEEN_EV = 1000
};

static const int pVal[7] = {0, PAWN_EV, KNIGHT_EV, BISHOP_EV, ROOK_EV, QUEEN_EV, 0};

//Mobility bonuses
static const int QueenMobility[28] = {
    -30, -20, -10, 0, 5, 10, 12, 15, 18, 20, 25, 30, 32, 35,
    40, 45, 50, 55, 57, 60, 63, 65, 70, 75, 80, 85, 90, 95
};
static const int RookMobility[15] = {-30, -20, -10, 0, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80};
static const int BishopMobility[14] = {-30, -10, 5, 15, 20, 25, 35, 40, 45, 50, 55, 60, 65, 70};
static const int KnightMobility[14] = {-50, -25, -10, -2, 5, 10, 15, 25};

//additional bonuses and penalties
static const int PassedPawnBonus[8] = {0, 0, 10, 20, 40, 80, 120, 0};
static const int DoubleBishopsBonus = S(30, 20);
static const int DoublePawnsPenalty = -15;
static const int IsolatedPawnPenalty = -5;
static const int RookOnOpenFileBonus = 20;
static const int RookOnPartOpenFileBonus = 10;
static const int PawnsUpperPiecesBonus = 3;
int distanceBonus[64][64];

//Hash eval
int IsolatedPawnsHash[256];

//global (using for speed-up)
int stage;

int KingDanger[100];

int fullEval(Board* board);
int materialEval(Board* board);
int psqtPieceEval(Board* board, U64 mask, const int* pstTable);
int mobilityAndKingDangerEval(Board* board, int color);
int pawnsEval(Board* board, int color);
int bishopsEval(Board* board);
int rooksEval(Board* board, int color);
int kingEval(Board* board, int color);
int attackCount(Board* board, int sq, int color);
int getPassedPawnBonus(int sq, int color);
int mateScore(int eval);
void initEval();
int stageGame(Board* board);
U8 horizontalScan(U64 bitboard);
int kingPsqtEval();
int baseEval(Board* board);
int kingDanger(int attacksCount);

#endif