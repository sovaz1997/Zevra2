#ifndef NNUE_H
#define NNUE_H

#include "board.h"

#define PERSPECTIVE_MASK 56
#define INPUTS_COUNT 768

// King-buckets: the feature transformer is selected by which board region the
// (own) king sits in. NNUE_KB enables 4 quadrant buckets; feature index becomes
// bucket*768 + base768. Default build has 1 bucket (== plain Chess768).
#ifdef NNUE_KB
#define NNUE_NUM_BUCKETS NNUE_KB   // build with -DNNUE_KB=4 / 8 / 16
#else
#define NNUE_NUM_BUCKETS 1
#endif
#define NNUE_FT_INPUTS (INPUTS_COUNT * NNUE_NUM_BUCKETS)
#ifndef INNER_LAYER_COUNT
#define INNER_LAYER_COUNT 256   // override at build with -DINNER_LAYER_COUNT=512
#endif
#define MAX_FEN_LENGTH 1000

// Quantisation / scaling, must match the bullet training config
// (nnue/trainer/src/main.rs).
#define NNUE_QA 255
#define NNUE_QB 64
#define NNUE_SCALE 400

// int8 inference of the L2 head (uint8 activations, int8 weights, vpdpbusd).
// Implies the L2 architecture; the f32 head is still read at load and quantised
// into the int8 tables below.
#if defined(NNUE_L2_I8) && !defined(NNUE_L2)
    #define NNUE_L2 NNUE_L2_I8
#endif

#ifdef NNUE_L2
// The non-linear head can saturate to very large logits on winning positions;
// cap the static eval well below the mate zone (MATE_LIMIT=29000) so it never
// gets confused with mate scores and destabilises the search.
#define NNUE_L2_EVAL_CAP 12000
#endif

#ifdef NNUE_L2_I8
#define NNUE_I8_ACT 127                          // activation uint8 scale: [0,1] -> [0,127]
#define NNUE_I8_W   64                           // weight int8 scale
#define NNUE_I8_MUL (NNUE_I8_ACT * NNUE_I8_W)    // 8128: product scale after one affine
#endif

// Architecture: (768 -> INNER)x2 -> 1, dual perspective, SCReLU.
// A single first-layer weight matrix (l0) feeds both perspectives; the
// stm/ntm accumulators differ only in their input feature indexing, which
// already matches bullet's Chess768 encoding (see getInputIndexOf).
struct NNUE {
    S16 inputs[NNUE_FT_INPUTS];
    S16 inputs_perspective[NNUE_FT_INPUTS];

    S16 featureWeights[NNUE_FT_INPUTS][INNER_LAYER_COUNT]; // l0w, QA-quantised
    S16 featureBias[INNER_LAYER_COUNT];                    // l0b, QA-quantised
    int wkBucket;   // current king bucket for the white (direct) accumulator
    int bkBucket;   // current king bucket for the black (perspective) accumulator
#ifdef NNUE_L2
    // Non-linear head (768->INNER)x2 -> NNUE_L2 -> 1, f32, computed in float.
    // l1w is input-major: l1w[i*NNUE_L2 + j] (bullet stores affine as
    // [input][output], same layout as the feature transformer l0w).
    float l1w[2 * INNER_LAYER_COUNT * NNUE_L2];
    float l1b[NNUE_L2];
    float l2w[NNUE_L2];
    float l2b;
  #ifdef NNUE_L2_I8
    // int8 tables derived from the f32 head at load time (see loadBulletNet).
    signed char l1w_i8[2 * INNER_LAYER_COUNT * NNUE_L2];  // input-major [i*L2+j]
    int         l1b_i32[NNUE_L2];
    signed char l2w_i8[NNUE_L2];
    int         l2b_i32;
  #endif
#else
    S16 outputWeights[2 * INNER_LAYER_COUNT];            // l1w, QB-quantised
    S16 outputBias;                                      // l1b, QA*QB-quantised
#endif

    S32 accumulators[INNER_LAYER_COUNT];             // white perspective
    S32 accumulators_perspective[INNER_LAYER_COUNT]; // black perspective
    int eval;
};

extern NNUE* nnue;
extern char fen[MAX_FEN_LENGTH];

int getInputIndexOf(int bucket, int color, int piece, int sq);
int nnueKingBucket(int sq);
void nnueUpdateBuckets(Board* board);   // refresh a perspective if its king changed bucket
void resetNNUE(NNUE* nnue);
void setDirectNNUEInput(NNUE* nnue, int index);
void resetDirectNNUEInput(NNUE* nnue, int index);
void setPerspectiveNNUEInput(NNUE* nnue, int index);
void resetPerspectiveNNUEInput(NNUE* nnue, int index);
void recalculateEval(NNUE* nnue, int color);
int loadBulletNet(const char* path);

#endif
