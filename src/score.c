#include "score.h"

int getScore(int score, int stageGame) {
    return MG(score) * stageGame / 98. + EG(score) * (1. - stageGame / 98.);
}

int getScore2(int mg, int eg, int stageGame) {
    return mg * stageGame / 98. + eg * (1. - stageGame / 98.);
}