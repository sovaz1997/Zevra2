#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

//Цвет фигуры
enum {
    WHITE = 0,
    BLACK = 1
};


//Названия фигур
enum {
    PAWN = 0,
    KNIGHT = 1,
    BISHOP = 2,
    ROOK = 3,
    QUEEN = 4,
    KING = 5,
    EMPTY = 6
};

//Периименование типов
typedef uint64_t U64;

//Структуры
typedef struct Board Board;

#endif