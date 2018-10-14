#ifndef BOARD_H
#define BOARD_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "types.h"
#include "bitboards.h"
#include "move.h"
#include "zobrist.h"

struct GameInfo {
    U64 moveHistory[8192];
    int moveCount;
};

struct Board {
    int ruleNumber;
    int enpassantSquare;
    U64 castling;
    int moveNumber;
    int color;
    U64 pieces[7];
    U64 colours[2];
    U8 squares[64];
    U64 key;
    GameInfo* gameInfo;
};

struct Undo {
    int ruleNumber;
    int enpassantSquare;
    U64 castling;
    U8 capturedPiece;
};

int isEqual(Board* b1, Board* b2);

void setFen(Board* board, char* fen);
void getFen(Board* board, char* fen);
void setMovesRange(Board* board, char* moves);
void clearBoard(Board* board);
void printBoard(Board* board);
void printBoardSplitter();
void setPiece(Board* board, int piece, int color, int square);
void clearPiece(Board* board, int square);
void movePiece(Board* board, int sq1, int sq2);
void squareToString(int square, char* str);
int stringToSquare(char* str);
U8 makePiece(int piece_type, int color);
void printPiece(U8 piece);
void makeMove(Board* board, U16 move, Undo* undo);
void unmakeMove(Board* board, U16 move, Undo* undo);
void setUndo(Board* board, Undo* undo, U8 capturedPiece);
void getUndo(Board* board, Undo* undo);
int attackedSquare(Board* board, int sq, int color);
int inCheck(Board* board, int color);

void addMoveToHist(Board* board);
void revertMoveFromHist(Board* board);

int isDraw(Board* board);

U8 firstAttacker(Board* board, U64 bitboard);
U8 lastAttacker(Board* board, U64 bitboard);

#define pieceType(piece) ((piece) >> 1)
#define pieceColor(piece) ((piece) & 1)

#endif