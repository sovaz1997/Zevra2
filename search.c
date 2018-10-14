#include "search.h"

void iterativeDeeping(Board* board, TimeManager tm) {
    SearchInfo searchInfo;
    char bestMove[6];
    
    resetSearchInfo(&searchInfo, tm);
    clearTT();
    startTimer(&searchInfo.timer);
    for(int i = 1; i <= tm.depth; ++i) {
        int eval = search(board, &searchInfo, -MATE_SCORE, MATE_SCORE, i, 0);

        if(searchInfo.abort) {
            break;
        }
        
        moveToString(searchInfo.bestMove, bestMove);
        printf("info depth %d nodes %d time %d ", i, searchInfo.nodesCount, getTime(&searchInfo.timer));
        printScore(eval);
        printf(" pv %s\n", bestMove);
        fflush(stdout);
    }

    printf("bestmove %s\n", bestMove);
    fflush(stdout);
}

int search(Board* board, SearchInfo* searchInfo, int alpha, int beta, int depth, int height) {
    if(searchInfo->abort) {
        return 0;
    }
    U64 keyPosition = board->key;
    Transposition* ttEntry = &tt[keyPosition & ttIndex];

    if(!depth) {
        return quiesceSearch(board, searchInfo, alpha, beta, height);
    }

    ++searchInfo->nodesCount;

    int root = (height ? 0 : 1);
    if(isDraw(board) && !root || searchInfo->abort) {
        return 0;
    }

    int extensions = inCheck(board, board->color);

    if(searchInfo->tm.searchType == FixedTime && depth >= 3) {
        if(getTime(&searchInfo->timer) >= searchInfo->tm.time) {
            searchInfo->abort = 1;
            return 0;
        }
    }

    if(ttEntry->evalType && ttEntry->depth >= depth && !root && ttEntry->key == keyPosition) {
        int score = ttEntry->eval;
        if(score > MATE_SCORE - 100) {
            score -= height;
        } else if(score < -MATE_SCORE + 100) {
            score += height;
        }

        if(ttEntry->evalType == lowerbound) {
            alpha = max(alpha, score);
        } else if(ttEntry->evalType == upperbound) {
            beta = min(beta, score);
        } else if(ttEntry->evalType == exact) {
            return score;
        }
    }

    movegen(board, moves[height]);
    moveOrdering(board, moves[height], searchInfo, height);

    U16* curMove = moves[height];
    Undo undo;

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

        int eval = -search(board, searchInfo, -beta, -alpha, depth - 1 + extensions, height + 1);        
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

            setTransposition(&new_tt, keyPosition, alpha, (alpha >= beta ? exact : lowerbound), depth, *curMove);
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
        setTransposition(&new_tt, keyPosition, alpha, upperbound, depth, 0);
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
    for(int attacker = 0; attacker < 7; ++attacker) {
        for(int victim = 0; victim < 7; ++victim) {
            int victimScore = 0;
            if(victim == QUEEN) {
                victimScore = 50;
            } else if(victim == ROOK) {
                victimScore = 40;
            } else if(victim == BISHOP) {
                victimScore = 30;
            } else if(victim == KNIGHT) {
                victimScore = 20;
            } else if(victim == PAWN) {
                victimScore = 10;
            }

            mvvLvaScores[attacker][victim] = victimScore - attacker;
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
    new_tr.eval = score;

    if(new_tr.depth > tr->depth) {
        if(new_tr.evalType == upperbound && tr->evalType != upperbound) {
            return;
        }
        *tr = new_tr;
    }
}