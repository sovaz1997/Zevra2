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