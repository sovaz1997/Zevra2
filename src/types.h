#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

#define min(a, b) (a < b ? a : b)
#define max(a, b) (a > b ? a : b)

//Color
enum {
    WHITE = 0,
    BLACK = 1
};

//Piece names
enum {
    PAWN = 1,
    KNIGHT = 2,
    BISHOP = 3,
    ROOK = 4,
    QUEEN = 5,
    KING = 6
};

//Direction
enum {
    UP = 0,
    DOWN = 1
};

//Search limits
enum {
    MAX_PLY = 128,
    MATE_SCORE = 30000,
    STAGE_N = 99,
};

//Piece chars
static const char PieceName[2][7] = {" PNBRQK", " pnbrqk"};

//Types
typedef uint64_t U64;
typedef uint32_t U32;
typedef uint16_t U16;
typedef uint8_t U8;
typedef int8_t S8;
typedef int16_t S16;
typedef int32_t S32;
typedef int64_t S64;

//Structures
typedef struct Board Board;
typedef struct Undo Undo;
typedef struct SearchInfo SearchInfo;
typedef struct GameInfo GameInfo;
typedef struct Timer Timer;
typedef struct TimeManager TimeManager;
typedef struct Transposition Transposition;
typedef struct SearchArgs SearchArgs;
typedef struct Option Option;
typedef struct Score Score;
typedef struct NNUE NNUE;

#endif