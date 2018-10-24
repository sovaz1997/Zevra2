#include "timemanager.h"

void startTimer(Timer* timer) {
    timer->startTime = clock();
}

U64 getTime(Timer* timer) {
    return (clock() - timer->startTime) / (CLOCKS_PER_SEC / 1000);
}

TimeManager createFixTimeTm(U64 millis) {
    TimeManager tm;
    tm.time = millis;
    tm.depth = MAX_PLY;
    tm.searchType = FixedTime;
    return tm;
}

TimeManager createFixDepthTm(int depth) {
    TimeManager tm;
    tm.depth = depth;
    tm.searchType = FidexDepth;
    return tm;
}

TimeManager createTournamentTm(Board* board, int wtime, int btime, int winc, int binc, int movesToGo) {
    TimeManager tm;
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
        tm->time = tm->tournamentTime[board->color] / tm->movesToGo + tm->tournamentInc[board->color] / 2;
    } else {
        int pieceCount = popcount(board->colours[WHITE] | board->colours[BLACK]);
        tm->time = tm->tournamentTime[board->color] / (40 - (32 - pieceCount)) + tm->tournamentInc[board->color] / 2;
    }
}

int testAbort(int time, TimeManager* tm) {
    return (tm->searchType == Tournament || tm->searchType == FixedTime) && time >= tm->time;
}