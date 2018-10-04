#ifndef BOARD_H
#define BOARD_H

#include "types.h"

struct Board {
    U64 pieces[6];
    U64 colours[2];
};

#endif