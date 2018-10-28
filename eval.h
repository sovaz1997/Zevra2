#ifndef EVAL_H
#define EVAL_H

#include "board.h"
#include "psqt.h"

//Веса фигур
enum figureWeights {
    PAWN_EV = 100,
    KNIGHT_EV = 300,
    BISHOP_EV = 330,
    ROOK_EV = 550,
    QUEEN_EV = 1000
};

static int pVal[7] = {0, PAWN_EV, KNIGHT_EV, BISHOP_EV, ROOK_EV, QUEEN_EV, 0};

//Бонусы мобильности
static int QueenMobility[28] = {
    -30, -20, -10, 0, 5, 10, 12, 15, 18, 20, 25, 30, 32, 35,
    40, 45, 50, 55, 57, 60, 63, 65, 70, 75, 80, 85, 90, 95
};
static int RookMobility[15] = {-30, -20, -10, 0, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80};
static int BishopMobility[14] = {-30, -10, 5, 15, 20, 25, 35, 40, 45, 50, 55, 60, 65, 70};
static int KnightMobility[14] = {-50, -25, -10, -2, 5, 10, 15, 25};

//Бонус проходных пешек
static int PassedPawnBonus[8] = {0, 0, 10, 20, 40, 80, 120, 0};

static int DoubleBishopsBonus = 30;
static int DoublePawnsPenalty = -15;

int distanceBonus[64][64];

static const int SafetyTable[100] = {
    0, 1, 2, 3, 5, 7, 9, 12, 15,
    18, 22, 26, 30, 35, 39, 44, 50, 56, 62,
    68, 75, 82, 85, 89, 97, 105, 113, 122, 131,
    140, 150, 169, 180, 191, 202, 213, 225, 237, 248,
    260, 272, 283, 295, 307, 319, 330, 342, 354, 366,
    377, 389, 401, 412, 424, 436, 448, 459, 471, 483,
    494, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500
};


int fullEval(Board* board);
int materialEval(Board* board);
int psqtEval(Board* board);
int psqtPieceEval(Board* board, U64 mask, const int* pstTable);
int mobilityEval(Board* board, int color);
int pawnsEval(Board* board, int color);
int bishopsEval(Board* board);
int rooksEval(Board* board, int color);
int kingSafety(Board* board, int color);
int kingEval(Board* board, int color);
int attackCount(Board* board, int sq, int color);
int getPassedPawnBonus(int sq, int color);
int mateScore(int eval);
int closeToMateScore(int eval);
int stageGame(Board* board);
void initEval();

#endif