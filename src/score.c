#include "score.h"

int getScore(int score, int stageGame) {
    return MG(score) * stageGame / 98. + EG(score) * (1. - stageGame / 98.);
}