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

// Tunable adaptive-TM coefficients (defaults set in initSearchParams; UCI-settable for SPSA).
extern int TmStabPct;     // soft limit shrinks by TmStabPct/100 per stable iteration (capped)
extern int TmChangeX100;  // multiply soft limit by this/100 when the best move just changed
extern int TmPanicX100;   // multiply soft limit by this/100 when the score dropped (fail-low)
extern int TmPanicDrop;   // score-drop threshold (cp) that triggers panic
extern int TmMaxMult;     // hard limit = optimum * TmMaxMult (capped at half the clock)

struct TimeManager {
    int searchType;
    int depth;
    int nodes;
    U64 time;
    U64 optimum;   // soft limit: stop starting new depths past this (adjusted by stability)
    U64 maximum;   // hard limit: never search longer than this (checked mid-search)
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
// Soft time correction: at the end of an iteration, should we stop starting new
// depths? Stable best move -> stop earlier; changed / score dropped -> allow more.
int shouldStopDeepening(TimeManager* tm, U64 elapsed, int depth, int stability, int eval, int prevEval);

#endif