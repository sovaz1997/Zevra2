#include "timemanager.h"

// Definitions for globals declared extern in timemanager.h
int TmStabPct;
int TmChangeX100;
int TmPanicX100;
int TmPanicDrop;
int TmMaxMult;

void startTimer(Timer* timer) {
    timer->startTime = clock();
}

U64 getTime(Timer* timer) {
    return (clock() - timer->startTime) / (CLOCKS_PER_SEC / 1000);
}

TimeManager createFixTimeTm(U64 millis) {
    TimeManager tm = initTM();
    tm.time = millis;
    tm.depth = MAX_PLY;
    tm.searchType = FixedTime;
    return tm;
}

TimeManager createFixDepthTm(int depth) {
    TimeManager tm = initTM();
    tm.depth = depth;
    tm.searchType = FixedDepth;
    return tm;
}

TimeManager initTM() {
    TimeManager tm;
    memset(&tm, 0, sizeof(TimeManager));
    return tm;
}

TimeManager createFixedNodesTm(int nodes) {
    TimeManager tm = initTM();
    tm.depth = MAX_PLY;
    tm.nodes = nodes;
    tm.searchType = FixedNodes;
    return tm;
}

TimeManager createTournamentTm(Board* board, int wtime, int btime, int winc, int binc, int movesToGo) {
    TimeManager tm = initTM();
    tm.tournamentTime[WHITE] = wtime;
    tm.tournamentTime[BLACK] = btime;
    tm.tournamentInc[WHITE] = winc;
    tm.tournamentInc[BLACK] = binc;
    tm.movesToGo = movesToGo;
    tm.searchType = Tournament;
    tm.depth = MAX_PLY;
    setTournamentTime(&tm, board);
    return tm;
}

void setTournamentTime(TimeManager* tm, Board* board) {
    int remaining = tm->tournamentTime[board->color];
    if(tm->movesToGo) {
        tm->time = remaining / (tm->movesToGo + 1) + tm->tournamentInc[board->color] / 2;
    } else {
        int pieceCount = popcount(board->colours[WHITE] | board->colours[BLACK]);
        tm->time = remaining / (40 - (32 - pieceCount)) + tm->tournamentInc[board->color] / 2;
    }
    // soft = base allocation; hard = up to TmMaxMult x, but never more than half the clock
    tm->optimum = tm->time;
    tm->maximum = tm->optimum * TmMaxMult;
    U64 hardCap = (U64)(remaining / 2);
    if (tm->maximum > hardCap) tm->maximum = hardCap;
    if (tm->maximum < tm->optimum) tm->maximum = tm->optimum;
}

int testAbort(U64 time, int nodesCount, TimeManager* tm) {
    return (tm->searchType == Tournament && time >= tm->maximum)
    | (tm->searchType == FixedTime && time >= tm->time)
    | (tm->searchType == FixedNodes && nodesCount >= tm->nodes);
}

int shouldStopDeepening(TimeManager* tm, U64 elapsed, int depth, int stability, int eval, int prevEval) {
    if (tm->searchType != Tournament || depth < 5)
        return 0;   // only adapt the tournament clock, and only after a few iterations
    double factor = 1.0 - (TmStabPct / 100.0) * min(stability, 6);   // stable -> less
    if (stability == 0) factor *= TmChangeX100 / 100.0;             // best move changed -> more
    if (eval < prevEval - TmPanicDrop) factor *= TmPanicX100 / 100.0; // score dropped -> more
    return elapsed >= (U64)(tm->optimum * factor);
}