#ifndef DATASET_H
#define DATASET_H

#include "board.h"
#include "uci.h"

typedef struct {
    char fen[256];
    int eval;
} Position;

typedef struct {
    Position positions[8192];
    int positionsCount;
} Game;

Game game;

char fen_for_save[256];
U16 moveList[256];

TimeManager createFixNodesTm(int nodes);

void resetGame(Game* game);
void addPosition(Game* game, char* fen, int eval);
void saveGameToFile(Game* game, FILE* file, double gameResult);


int getMovesCount(Board* board);
void makeRandomMove(Board* board);
void runGame(Board* board, FILE* file);
void createDataset(Board* board, int gamesCount, int seed, char* fileName, char* logFile);

#endif