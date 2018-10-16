#ifndef TIMEMANAGER_H
#define TIMEMANAGE_C

#include <time.h>
#include "types.h"
#include "board.h"

//Тип поиска
enum {
    FixedTime = 0,
    FidexDepth = 1,
    Tournament = 2
};

struct Timer {
    clock_t startTime;
};

struct TimeManager {
    int searchType;
    int depth;
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
void setTournamentTime(TimeManager* tm, Board* board);
int testAbort(int time, TimeManager* tm);

#endif