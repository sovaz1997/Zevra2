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

extern int pVal[7];

//Бонусы мобильности
extern int QueenMobility[28];
extern int RookMobility[15];
extern int BishopMobility[14];
extern int KnightMobility[14];

//Бонус проходных пешек
extern int PassedPawnBonus[8];

//Бонус 2-х слонов
extern int DoubleBishopsBonus;

extern int DoublePawnsPenalty;

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
int kingSafety(Board* board, int color);
int attackCount(Board* board, int sq, int color);
int getPassedPawnBonus(int sq, int color);
int mateScore(int eval);
int closeToMateScore(int eval);
int stageGame(Board* board);

#endif