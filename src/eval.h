#ifndef EVAL_H
#define EVAL_H

#include "board.h"
#include "psqt.h"
#include "score.h"

enum {
  QUEEN_MOBILITY_N = 28,
    ROOK_MOBILITY_N = 15,
    BISHOP_MOBILITY_N = 14,
    KNIGHT_MOBILITY_N = 9,
};

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
extern int QueenMobilityMG[QUEEN_MOBILITY_N];
extern int RookMobilityMG[ROOK_MOBILITY_N];
extern int BishopMobilityMG[BISHOP_MOBILITY_N];
extern int KnightMobilityMG[KNIGHT_MOBILITY_N];

extern int QueenMobilityEG[QUEEN_MOBILITY_N];
extern int RookMobilityEG[ROOK_MOBILITY_N];
extern int BishopMobilityEG[BISHOP_MOBILITY_N];
extern int KnightMobilityEG[KNIGHT_MOBILITY_N];

int QueenMobility[STAGE_N][QUEEN_MOBILITY_N];
int RookMobility[STAGE_N][ROOK_MOBILITY_N];
int BishopMobility[STAGE_N][BISHOP_MOBILITY_N];
int KnightMobility[STAGE_N][KNIGHT_MOBILITY_N];

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

int psqtPieceEval(Board *board, U64 mask, int pieceType);

int mobilityAndKingDangerEval(Board *board, int color);

int pawnsEval(Board *board, int color);

int bishopsEval(Board *board);

int rooksEval(Board *board, int color);

int getPassedPawnBonus(int sq, int color);

int mateScore(int eval);

void initEval();

void destroyEval();

int stageGame(Board *board);

U8 horizontalScan(U64 bitboard);

int psqtEval(Board* board);

int kingDanger(int attacksCount);

void initDependencyEval();
void initStagedPSQT(int st);
void initDependencyStagedEval(int st);

#endif