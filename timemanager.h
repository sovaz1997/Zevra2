#ifndef TIMEMANAGER_H
#define TIMEMANAGE_C

#include <time.h>
#include "types.h"

//Тип поиска
enum {
    FixedTime = 0,
    FidexDepth = 1
};

struct Timer {
    clock_t startTime;
};

struct TimeManager {
    int searchType;
    int depth;
    U64 time;
};

void startTimer(Timer* timer);
U64 getTime(Timer* timer);

TimeManager createFixTimeTm(U64 millis);
TimeManager createFixDepthTm(int depth);

#endif