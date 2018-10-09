#ifndef BOARD_H
#define BOARD_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "types.h"
#include "bitboards.h"
#include "move.h"

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

struct Undo {
    int ruleNumber;
    int enpassantSquare;
    int castling;
    U8 capturedPiece;
};

void setFen(Board* board, char* fen);
void getFen(Board* board, char* fen);
void clearBoard(Board* board);
void printBoard(Board* board);
void printBoardSplitter();
void setPiece(Board* board, int piece, int color, int square);
U8 clearPiece(Board* board, int square);
void movePiece(Board* board, int sq1, int sq2);
void squareToString(int square, char* str);
int stringToSquare(char* str);
U8 makePiece(int piece_type, int color);
void printPiece(U8 piece);
void makeMove(Board* board, U16 move, Undo* undo);
void unmakeMove(Board* board, U16 move, Undo* undo);
void setUndo(Board* board, Undo* undo, U8 capturedPiece);
void getUndo(Board* board, Undo* undo);

#define pieceType(piece) ((piece) >> 1)
#define pieceColor(piece) ((piece) & 1)

#endif