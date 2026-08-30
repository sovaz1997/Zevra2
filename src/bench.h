#ifndef BENCH_H
#define BENCH_H

#include "board.h"

// Deterministic run over a fixed set of positions at a fixed depth.
// Prints the total node count (signature) and nps.
// Two builds with the same signature have functionally identical search.
void benchmark(Board* board, int depth);

#endif
