#include "psqt.h"

//int allPST[0] = pawnPST;
//int allPST[1] = pawnPST;
//int allPST[2] = knightPST;
//int allPST[3] = bishopPST;
//int allPST[4] = rookPST;
//int allPST[5] = queenPST;

int initPSQT() {
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 64; j++) {
            if (i == 0) {
                allPST[i][j] = pawnPST[j];
            }
            else if (i == 1) {
                allPST[i][j] = pawnPST[j];
            }
            else if (i == 2) {
                allPST[i][j] = knightPST[j];
            }
            else if (i == 3) {
                allPST[i][j] = bishopPST[j];
            }
            else if (i == 4) {
                allPST[i][j] = rookPST[j];
            }
            else if (i == 5) {
                allPST[i][j] = queenPST[j];
            }
        }
    }
}