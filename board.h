#ifndef BOARD_H
#define BOARD_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "types.h"
#include "bitboards.h"

struct Board {
    int moveNumber;
    int ruleNumber;
    int color;
    int enpassantSquare;
    int castling;
    U64 pieces[7];
    U64 colours[2];
    U8 squares[64];
};

void setFen(Board* board, char* fen);
void getFen(Board* board, char* fen);
void clearBoard(Board* board);
void printBoard(Board* board);
void printBoardSplitter();
void setPiece(Board* board, int piece, int color, int square);
void squareToString(int square, char* str);
int stringToSquare(char* str);
uint8_t makePiece(int piece_type, int color);
int pieceColor(uint8_t piece);
int piceType(uint8_t piece);
void printPiece(uint8_t piece);

#endif