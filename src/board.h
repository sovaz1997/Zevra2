#ifndef BOARD_H
#define BOARD_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <ctype.h>
#include "types.h"
#include "bitboards.h"
#include "move.h"
#include "zobrist.h"
#include "eval.h"

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

void setFen(Board* board, char* fen);
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
U64 attacksTo(Board* board, int sq);
int inCheck(Board* board, int color);

void addMoveToHist(Board* board);
void revertMoveFromHist(Board* board);

void makeNullMove(Board* board);
void unmakeNullMove(Board* board);

int isDraw(Board* board);
int repeatCount(Board* board);

#define pieceType(piece) ((piece) >> 1)
#define pieceColor(piece) ((piece) & 1)

//Position signs
int havePromotionPawn(Board* board);
int haveNoPawnMaterial(Board* board);

//Static exchange evaluation functions
int see(Board* board, int toSq, U8 taget, int fromSq, U8 aPiece);
U64 getLeastValuablePiece(Board* board, U64 attadef, int side, U8* piece);
U64 considerXrays(Board* board, U64 occu, U64 attackdef, int sq);

#endif