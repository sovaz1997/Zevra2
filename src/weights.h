#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "nnue.h"

#define FC1_SIZE INNER_LAYER_COUNT * INPUTS_COUNT
#define FC2_SIZE 2 * INNER_LAYER_COUNT

extern const double fc1_us[FC1_SIZE];
extern const double fc1_them[FC1_SIZE];
extern const double fc2[FC2_SIZE];

#endif