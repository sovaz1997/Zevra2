#ifndef TIMEMANAGER_H
#define TIMEMANAGER_H

#include <time.h>
#include "types.h"
#include "board.h"

//Search type
enum {
    FixedTime = 0,
    FixedDepth = 1,
    Tournament = 2,
    FixedNodes = 3
};

struct Timer {
    clock_t startTime;
};

struct TimeManager {
    int searchType;
    int depth;
    int nodes;
    U64 time;
    int tournamentTime[2];
    int tournamentInc[2];
    int movesToGo;
};

void startTimer(Timer* timer);
U64 getTime(Timer* timer);

TimeManager createFixTimeTm(U64 millis);
TimeManager createFixDepthTm(int depth);
TimeManager createTournamentTm(Board* board, int wtime, int btime, int winc, int binc, int movesToGo);
TimeManager createFixedNodesTm(int depth);
TimeManager initTM();
void setTournamentTime(TimeManager* tm, Board* board);
int testAbort(U64 time, int nodesCount, TimeManager* tm);
void replaceTranspositionEntry(Transposition* addr, TranspositionEntity* newEntry);

#endif