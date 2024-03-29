#include "timemanager.h"

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
    if(tm->movesToGo) {
        tm->time = tm->tournamentTime[board->color] / (tm->movesToGo + 1) + tm->tournamentInc[board->color] / 2;
    } else {
        int pieceCount = popcount(board->colours[WHITE] | board->colours[BLACK]);
        tm->time = tm->tournamentTime[board->color] / (40 - (32 - pieceCount)) + tm->tournamentInc[board->color] / 2;
    }
}

int testAbort(U64 time, int nodesCount, TimeManager* tm) {
    return ((tm->searchType == Tournament || tm->searchType == FixedTime) && time >= tm->time)
    | (tm->searchType == FixedNodes && nodesCount >= tm->nodes);
}