#ifndef EVAL_H
#define EVAL_H

#include "board.h"
#include "psqt.h"
#include "score.h"

//Piece weights
extern int PAWN_EV_MG;
extern int KNIGHT_EV_MG;
extern int BISHOP_EV_MG;
extern int ROOK_EV_MG;
extern int QUEEN_EV_MG;

extern int PAWN_EV_EG;
extern int KNIGHT_EV_EG;
extern int BISHOP_EV_EG;
extern int ROOK_EV_EG;
extern int QUEEN_EV_EG;

int* PAWN_EVAL;
int* KNIGHT_EVAL;
int* BISHOP_EVAL;
int* ROOK_EVAL;
int* QUEEN_EVAL;

int pVal(Board* b, int n);

//Mobility bonuses
extern int QueenMobility[28];
extern int RookMobility[15];
extern int BishopMobility[14];
extern int KnightMobility[8];

//additional bonuses and penalties
extern int PassedPawnBonus[8];
extern int DoublePawnsPenalty;
extern int IsolatedPawnPenalty;
extern int RookOnOpenFileBonus;
extern int RookOnPartOpenFileBonus;
int distanceBonus[64][64];

extern int DoubleBishopsBonusMG;
extern int DoubleBishopsBonusEG;

//Hash eval
int IsolatedPawnsHash[256];

//global (using for speed-up)
int stage;

int KingDanger[100];

int KingDangerFactor;

int fullEval(Board *board);

int DoubleBishopsBonus();
int materialEval(Board* board);

int psqtPieceEval(Board *board, U64 mask, const int *pstTable);

int mobilityAndKingDangerEval(Board *board, int color);

int pawnsEval(Board *board, int color);

int bishopsEval(Board *board);

int rooksEval(Board *board, int color);

int kingEval(Board *board, int color);

int attackCount(Board *board, int sq, int color);

int getPassedPawnBonus(int sq, int color);

int mateScore(int eval);

void initEval();

void destroyEval();

int stageGame(Board *board);

U8 horizontalScan(U64 bitboard);

int kingPsqtEval(Board* board);

int kingDanger(int attacksCount);

void initDependencyEval();

#endif