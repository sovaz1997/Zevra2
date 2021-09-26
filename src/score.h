#ifndef SCORE_H
#define SCORE_H

#include "types.h"

int getScore(int score, int stageGame);
int getScore2(int mg, int eg, int stageGame);

#define CreateScore(mg, eg) (((mg) << 16) + (eg))
#define S(mg, eg) CreateScore((mg), (eg))
#define MG(score) (S16)(U16)((score) >> 16)
#define EG(score) (S16)(U16)(score & 0xffff)

#endif