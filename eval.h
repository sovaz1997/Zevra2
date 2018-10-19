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


int fullEval(Board* board);
int materialEval(Board* board);
int psqtEval(Board* board);
int psqtPieceEval(Board* board, U64 mask, const int* pstTable);
int mobilityEval(Board* board, int color);
int pawnsEval(Board* board, int color);
int bishopsEval(Board* board);
int getPassedPawnBonus(int sq, int color);
int mateScore(int eval);

#endif