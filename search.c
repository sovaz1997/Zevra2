#include "search.h"

void iterativeDeeping(Board* board, TimeManager tm) {
    ++ttAge;
    SearchInfo searchInfo;
    char bestMove[6];

    resetSearchInfo(&searchInfo, tm);
    startTimer(&searchInfo.timer);
    for(int i = 1; i <= tm.depth; ++i) {
        int eval = search(board, &searchInfo, -MATE_SCORE, MATE_SCORE, i, 0);
        
        if(searchInfo.abort && i > 1) {
            break;
        }
        
        moveToString(searchInfo.bestMove, bestMove);
        printf("info depth %d nodes %d time %d ", i, searchInfo.nodesCount, getTime(&searchInfo.timer));
        printScore(eval);
        printf(" pv ", bestMove);
        printPV(board, i);
        printf("\n");
        fflush(stdout);
    }

    printf("info nodes %llu time %d\n", searchInfo.nodesCount, getTime(&searchInfo.timer));
    printf("bestmove %s\n", bestMove);
    fflush(stdout);
}

int search(Board* board, SearchInfo* searchInfo, int alpha, int beta, int depth, int height) {
    if(searchInfo->abort) {
        return 0;
    }

    U64 keyPosition = board->key;
    Transposition* ttEntry = &tt[keyPosition & ttIndex];

    int root = (height ? 0 : 1);
    int pvNode = (beta - alpha > 1);

    if(isDraw(board) && !root || searchInfo->abort) {
        return 0;
    }

    int extensions = !!inCheck(board, board->color);

    if(searchInfo->tm.searchType == FixedTime && depth >= 3) {
        if(getTime(&searchInfo->timer) >= searchInfo->tm.time) {
            searchInfo->abort = 1;
            return 0;
        }
    }

    Undo undo;

    ++searchInfo->nodesCount;

    if(ttEntry->evalType && ttEntry->depth >= depth && !root && ttEntry->key == keyPosition && depth > 1) {
        int score = ttEntry->eval;
        if(score > MATE_SCORE - 100) {
            score -= height;
        } else if(score < -MATE_SCORE + 100) {
            score += height;
        }

        if(ttEntry->evalType == lowerbound && score >= beta) {
            return score;
        } else if(ttEntry->evalType == upperbound && score <= alpha) {
            return score;
        } else if(ttEntry->evalType == exact) {
            return score;
        }
    }

    if(depth <= 0 && !extensions) {
        return quiesceSearch(board, searchInfo, alpha, beta, height);
    }

    //Null Move pruning
    
    int R = 2 + depth / 6;
    int staticEval = fullEval(board);
    if(!extensions && !searchInfo->nullMoveSearch && depth > R && (staticEval >= beta || depth < 4)) {
        makeNullMove(board);
        searchInfo->nullMoveSearch = 1;

        int eval = -search(board, searchInfo, -beta, -beta + 1, depth - 1 - R, height + 1);

        searchInfo->nullMoveSearch = 0;
        unmakeNullMove(board);

        if(eval >= beta) {
            return beta;
        }
    }

    movegen(board, moves[height]);
    moveOrdering(board, moves[height], searchInfo, height);

    U16* curMove = moves[height];

    int movesCount = 0;

    Transposition new_tt;

    int oldAlpha = alpha;
    while(*curMove) {
        makeMove(board, *curMove, &undo);

        if(inCheck(board, !board->color)) {
            unmakeMove(board, *curMove, &undo);
            ++curMove;
            continue;
        }
        
        ++movesCount;

        int reductions = lmr[min(depth, MAX_PLY - 1)][min(63, movesCount)];
        int quiteMove = (searchInfo->killer[board->color][height] != *curMove && !inCheck(board, board->color) && !undo.capturedPiece && !extensions);

        int eval;
        if(movesCount == 1) {
            eval = -search(board, searchInfo, -beta, -alpha, depth - 1 + extensions, height + 1);
        } else {
            if(movesCount >= 3 && quiteMove && !pvNode) {
                eval = -search(board, searchInfo, -alpha - 1, -alpha, depth - 1 + extensions - reductions, height + 1);
                if(eval > alpha && eval < beta) {
                    eval = -search(board, searchInfo, -beta, -alpha, depth - 1 + extensions, height + 1);
                }
            } else {
                eval = -search(board, searchInfo, -alpha - 1, -alpha, depth - 1 + extensions, height + 1);

                if(eval > alpha && eval < beta) {
                    eval = -search(board, searchInfo, -beta, -alpha, depth - 1 + extensions, height + 1);
                }
            }
        }

        unmakeMove(board, *curMove, &undo);
        
        if(root) {
            char moveStr[6];
            moveToString(*curMove, moveStr);
            printf("info move %s currmovenumber %d\n", moveStr, movesCount);
            fflush(stdout);
        }
        
        if(eval > alpha) {
            alpha = eval;
            if(root) {
                searchInfo->bestMove = *curMove;
            }

            if(!undo.capturedPiece) {
                searchInfo->killer[board->color][height] = *curMove;
                searchInfo->history[MoveFrom(*curMove)][MoveTo(*curMove)] += (depth * depth);
            }

            setTransposition(&new_tt, keyPosition, alpha, (alpha >= beta ? lowerbound : exact), depth, *curMove, ttAge);
        }
        if(alpha >= beta) {
            break;
        }
        ++curMove;
    }

    if(searchInfo->abort) {
            return 0;
    }

    if(oldAlpha == alpha) {
        setTransposition(&new_tt, keyPosition, alpha, upperbound, depth, 0, ttAge);
    }

    replaceTransposition(ttEntry, new_tt, height);

    if(!movesCount) {
        if(inCheck(board, board->color)) {
            return -MATE_SCORE + height;
        } else {
            return 0;
        }
    }

    return alpha;
}

int quiesceSearch(Board* board, SearchInfo* searchInfo, int alpha, int beta, int height) {
    if(searchInfo->tm.searchType == FixedTime) {
        if(getTime(&searchInfo->timer) >= searchInfo->tm.time) {
            searchInfo->abort = 1;
            return 0;
        }
    }

    if(searchInfo->abort) {
            return 0;
    }

    ++searchInfo->nodesCount;
    
    int val = fullEval(board);
    if(val >= beta) {
        return beta;
    }
    if(alpha < val) {
        alpha = val;
    }

    attackgen(board, moves[height]);
    moveOrdering(board, moves[height], searchInfo, height);
    U16* curMove = moves[height];
    Undo undo;
    while(*curMove) {
        if(searchInfo->abort) {
            return 0;
        }

        makeMove(board, *curMove, &undo);
    
        if(inCheck(board, !board->color)) {
            unmakeMove(board, *curMove, &undo);
            ++curMove;
            continue;
        }

        int score = -quiesceSearch(board, searchInfo, -beta, -alpha, height + 1);

        unmakeMove(board, *curMove, &undo);
        if(score >= beta) {
            return beta;
        }
        if(score > alpha) {
           alpha = score;
        }
        ++curMove;
    }

    if(searchInfo->abort) {
            return 0;
    }

    return alpha;
}

U64 perftTest(Board* board, int depth, int height) {
    if(!depth) {
        return 1;
    }


    movegen(board, moves[height]);

    U64 result = 0;
    U16* curMove = moves[height];
    Undo undo;
    while(*curMove) {
        makeMove(board, *curMove, &undo);

        U64 count = 0;
        if(!inCheck(board, !board->color)) {
            count = perftTest(board, depth - 1, height + 1);

            if(!height) {
                char mv[6];
                moveToString(*curMove, mv);
                for(int i = 0; i < height; ++i) {
                    printf(" ");
                }
                printf("%s: %llu\n", mv, count);
            }
        }

        result += count;

        unmakeMove(board, *curMove, &undo);

        ++curMove;
    }

    return result;
}

void perft(Board* board, int depth) {
    for(int i = 1; i <= depth; ++i) {
        clock_t start = clock();
        U64 nodes = perftTest(board, i, 0);
        clock_t end = clock();
        if(!(end - start)) {
            end = start + 1;
        }
        
        printf("Perft %d: %llu; speed: %d\n", i, nodes, nodes / (end - start));
    }
}

void moveOrdering(Board* board, U16* moves, SearchInfo* searchInfo, int height) {
    U16* ptr = moves;
    U16 hashMove = tt[board->key & ttIndex].move;
    int i;

    for(i = 0; *ptr; ++i) {
        movePrice[i] = 0;
        U16 toPiece = pieceType(board->squares[MoveTo(*ptr)]);
        U16 fromPiece = pieceType(board->squares[MoveFrom(*ptr)]);
        
        if(hashMove == *ptr) {
            movePrice[i] = 1000000000;
        } else if(toPiece) {
            movePrice[i] = mvvLvaScores[fromPiece][toPiece] * 1000000;
        } else if(searchInfo->killer[board->color][height] == *ptr) {
            movePrice[i] = 100000;
        } else {
            movePrice[i] = searchInfo->history[MoveFrom(*ptr)][MoveTo(*ptr)];
        }

        if(searchInfo->bestMove == *ptr && !height) {
            movePrice[i] = 1000000000;
        } 

        ++ptr;
    }

    sort(moves, i);
}

void sort(U16* moves, int count) {
    int i, j, key;
    U16 keyMove;
    for (i = 1; i < count; i++)  { 
        key = movePrice[i];
        keyMove = moves[i];
        j = i - 1; 
    
        while (j >= 0 && movePrice[j] < key) { 
            movePrice[j + 1] = movePrice[j];
            moves[j + 1] = moves[j];
            --j;
        } 
        movePrice[j + 1] = key;
        moves[j + 1] = keyMove;
    }
}

void initSearch() {
    for(int attacker = 1; attacker < 7; ++attacker) {
        for(int victim = 1; victim < 7; ++victim) {
            int victimScore = 0;
            mvvLvaScores[attacker][victim] = 64 * victim - attacker;
        }
    }

    for(int i = 0; i < MAX_PLY; ++i) {
        for(int j = 0; j < 64; ++j) {
            lmr[i][j]  = 0.75 + log(i) * log(j) / 2.25;
        }   
    }
}

void resetSearchInfo(SearchInfo* info, TimeManager tm) {
    memset(info, 0, sizeof(SearchInfo));
    info->tm = tm;
    memset(info->history, 0, 64 * 64);
}

void replaceTransposition(Transposition* tr, Transposition new_tr, int height) {
    int score = new_tr.eval;
    if(score > MATE_SCORE - 100) {
        score += height;
    } else if(score < -MATE_SCORE + 100) {
        score -= height;
    }

    if(tr->age + 5 < ttAge) {
        *tr = new_tr;
        return;
    }

    if(new_tr.depth > tr->depth) {
        new_tr.eval = score;
        if(new_tr.evalType == upperbound && tr->evalType != upperbound) {
            return;
        }
        *tr = new_tr;
    }
}