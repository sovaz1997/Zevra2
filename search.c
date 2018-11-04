#include "search.h"


int FutilityMargin[7] = {0, 50, 200, 250, 350, 500, 700};

void* go(void* thread_data) {
    SearchArgs* args = (SearchArgs*)thread_data;
    iterativeDeeping(args->board, args->tm);
    SEARCH_COMPLETE = 1;
}

void iterativeDeeping(Board* board, TimeManager tm) {
    ++ttAge;
    SearchInfo searchInfo;
    char bestMove[6];

    resetSearchInfo(&searchInfo, tm);
    startTimer(&searchInfo.timer);
    int eval = 0;
    for(int i = 1; i <= tm.depth; ++i) {
        eval = search(board, &searchInfo, -MATE_SCORE, MATE_SCORE, i, 0);//aspirationWindow(board, &searchInfo, i, eval);
        
        if(ABORT && i > 1) {
            break;
        }
        moveToString(searchInfo.bestMove, bestMove);

        U64 searchTime = getTime(&searchInfo.timer);
        int speed = (searchTime < 1 ? 0 : (searchInfo.nodesCount / (searchTime / 1000.)));
        int hashfull = (double)ttFilledSize  / (double)ttSize * 1000;

        printf("info depth %d seldepth %d nodes %llu time %llu nps %d hashfull %d ", i, searchInfo.selDepth, searchInfo.nodesCount, searchTime, speed, hashfull);
        printScore(eval);
        printf(" pv ");
        printPV(board, i, searchInfo.bestMove);
        printf("\n");
        fflush(stdout);
    }


    printf("info nodes %llu time %llu\n", searchInfo.nodesCount, getTime(&searchInfo.timer));
    printf("bestmove %s\n", bestMove);
    fflush(stdout);
}

int aspirationWindow(Board* board, SearchInfo* searchInfo, int depth, int score) {
    int delta = 15;
    int alpha = max(-MATE_SCORE, score - delta);
    int beta = min(MATE_SCORE, score + delta);

    if(depth <= 5) {
        return search(board, searchInfo, -MATE_SCORE, MATE_SCORE, depth, 0);
    }

    char bestMove[6];

    int f = score;
    printf("%d %d\n", alpha, beta);
    while(abs(f) < MATE_SCORE - 1) {
         f = search(board, searchInfo, alpha, beta, depth, 0);

        U64 searchTime = getTime(&searchInfo->timer);
        int speed = (searchTime < 1 ? 0 : (searchInfo->nodesCount / (searchTime / 1000.)));
        int hashfull = (double)ttFilledSize  / (double)ttSize * 1000;

        moveToString(searchInfo->bestMove, bestMove);

        if(ABORT) {
            break;
        }

        if(f > alpha && f < beta) {
            printf("info depth %d nodes %llu time %llu nps %d hashfull %d ", depth, searchInfo->nodesCount, searchTime, speed, hashfull);
            printScore(f);
            printf(" pv ");
            printPV(board, depth, searchInfo->bestMove);
            printf("\n");
            fflush(stdout);

            return f;
        }

        if(f <= alpha) {
            beta = (alpha + beta) / 2;
            alpha = max(-MATE_SCORE, alpha - delta);

            printf("info depth %d nodes %llu time %llu nps %d hashfull %d ", depth, searchInfo->nodesCount, searchTime, speed, hashfull);
            printScore(f);
            printf(" upperbound pv ");
            printPV(board, depth, searchInfo->bestMove);
            printf("\n");
            fflush(stdout);
        }

        if(f >= beta) {
            beta = min(MATE_SCORE, beta + delta);

            printf("info depth %d nodes %llu time %llu nps %d hashfull %d ", depth, searchInfo->nodesCount, searchTime, speed, hashfull);
            printScore(f);
            printf(" lowerbound pv ");
            printPV(board, depth, searchInfo->bestMove);
            printf("\n");
            fflush(stdout);
        }

        delta += delta / 2;
    }
}

int search(Board* board, SearchInfo* searchInfo, int alpha, int beta, int depth, int height) {
    ++searchInfo->nodesCount;
    if(ABORT) {
        return 0;
    }

    if(depth < 0) {
        depth = 0;
    }

    U64 keyPosition = board->key;
    Transposition* ttEntry = &tt[keyPosition & ttIndex];

    int root = (height ? 0 : 1);
    int pvNode = (beta - alpha > 1);

    if(isDraw(board) && !root || ABORT) {
        return 0;
    }

    int weInCheck = !!(inCheck(board, board->color));

    if(depth >= 3 && testAbort(getTime(&searchInfo->timer), &searchInfo->tm)) {
        setAbort(1);
        return 0;
    }

    Undo undo;

    if(ttEntry->evalType && ttEntry->depth >= depth && !root /*&& depth > 1*/ && ttEntry->key == keyPosition) {
        int score = ttEntry->eval;
        if(score > MATE_SCORE - 100) {
            score -= height;
        } else if(score < -MATE_SCORE + 100) {
            score += height;
        }

        if(ttEntry->evalType == lowerbound && score >= beta && !mateScore(score)) {
            return score;
        } else if(ttEntry->evalType == upperbound && score <= alpha && !mateScore(score)) {
            return score;
        } else if(ttEntry->evalType == exact) {
            return score;
        }
    }

    if((depth <= 0 && !weInCheck) || height >= MAX_PLY - 1) {
        return quiesceSearch(board, searchInfo, alpha, beta, height);
    }

    //Null Move pruning
    
    int R = 2 + depth / 6;
    int staticEval = fullEval(board);
    if(NullMovePruningAllow && !pvNode && haveNoPawnMaterial(board) && !weInCheck && !root && !searchInfo->nullMoveSearch && depth > R && (staticEval >= beta || depth <= 4)) {
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
    moveOrdering(board, moves[height], searchInfo, height, depth);

    U16* curMove = moves[height];

    int movesCount = 0;

    Transposition new_tt;

    int oldAlpha = alpha;
    while(*curMove) {
        int seeScore = 0;
        int nextDepth = depth - 1;
        if(board->squares[MoveTo(*curMove)]) {
            seeScore = see(board, MoveTo(*curMove), board->squares[MoveTo(*curMove)], MoveFrom(*curMove), board->squares[MoveFrom(*curMove)]);
        }

        makeMove(board, *curMove, &undo);

        if(inCheck(board, !board->color)) {
            unmakeMove(board, *curMove, &undo);
            ++curMove;
            continue;
        }
        
        ++movesCount;

        int extensions = inCheck(board, board->color);
        int goodMove = (searchInfo->killer[board->color][depth] == *curMove
        || searchInfo->secondKiller[board->color][depth] == *curMove
        || extensions || MoveType(*curMove) == PROMOTION_MOVE
        );
        int quiteMove = (!goodMove && !undo.capturedPiece);

        if(root && depth > 12) {
            char moveStr[6];
            moveToString(*curMove, moveStr);
            printf("info currmove %s currmovenumber %d\n", moveStr, movesCount);
            fflush(stdout);
        }

        //Fulility pruning
        if(depth < 7 && !goodMove && !root && FutilityPruningAllow) {
            if(staticEval + FutilityMargin[depth] + pVal[pieceType(undo.capturedPiece)] <= alpha) {
                unmakeMove(board, *curMove, &undo);
                ++curMove;
                continue;
            }
        }

        int reductions = lmr[min(depth, MAX_PLY - 1)][min(63, movesCount)] * quiteMove + (seeScore < 0);
        int historyReduced = 0;

        //History pruning
        if(HistoryPruningAllow && !pvNode && !extensions && !goodMove && depth >= 7 && movePrice[height][movesCount - 1] >= 0 && movePrice[height][movesCount - 1] <= 20000) {
            --nextDepth;
            historyReduced = 1;
        }

        int eval;
        if(movesCount == 1) {
            eval = -search(board, searchInfo, -beta, -alpha, nextDepth + extensions, height + 1);
        } else {
            if(LmrPruningAllow && movesCount >= 3 && quiteMove && !pvNode) {
                eval = -search(board, searchInfo, -alpha - 1, -alpha, nextDepth + extensions - reductions, height + 1);
                if(eval > alpha) {
                    eval = -search(board, searchInfo, -alpha - 1, -alpha, nextDepth + extensions, height + 1);
                    if(eval > alpha && eval < beta) {
                        eval = -search(board, searchInfo, -beta, -alpha, nextDepth + extensions, height + 1);
                    }
                }
            } else {
                eval = -search(board, searchInfo, -alpha - 1, -alpha, nextDepth + extensions, height + 1);
    
                if(eval > alpha && eval < beta) {
                    eval = -search(board, searchInfo, -beta, -alpha, nextDepth + extensions, height + 1);
                }
            }
        }

        if(HistoryPruningAllow &&  historyReduced && eval >= beta && depth >= 3) {
            ++nextDepth;
            eval = -search(board, searchInfo, -beta, -alpha, nextDepth + extensions, height + 1);
        }

        unmakeMove(board, *curMove, &undo);
        
        if(eval > alpha) {
            alpha = eval;
            if(root) {
                searchInfo->bestMove = *curMove;
            }

            setTransposition(&new_tt, keyPosition, alpha, (alpha >= beta ? lowerbound : exact), depth, *curMove, ttAge);
        }
        if(alpha >= beta) {
            if(!undo.capturedPiece) {
                if(searchInfo->killer[board->color][depth]) {
                    searchInfo->secondKiller[board->color][height] = searchInfo->killer[board->color][depth];
                }
                
                searchInfo->killer[board->color][depth] = *curMove;
                history[board->color][MoveFrom(*curMove)][MoveTo(*curMove)] += (depth * depth);
            }

            break;
        }
        ++curMove;
    }

    if(ABORT) {
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
    searchInfo->selDepth = max(searchInfo->selDepth, height);
    if(height >= MAX_PLY - 1) {
        return fullEval(board);
    }
    
    if(testAbort(getTime(&searchInfo->timer), &searchInfo->tm)) {
        setAbort(1);
        return 0;
    }

    if(ABORT) {
        return 0;
    }
    
    int val = fullEval(board);
    if(val >= beta) {
        return beta;
    }

    int delta = QUEEN_EV;
    if(havePromotionPawn(board)) {
        delta += (QUEEN_EV - 200);
    }

    if(val < alpha - delta) {
        return val;
    }

    if(alpha < val) {
        alpha = val;
    }

    attackgen(board, moves[height]);
    moveOrdering(board, moves[height], searchInfo, height, 0);
    U16* curMove = moves[height];
    Undo undo;
    while(*curMove) {
        if(ABORT) {
            return 0;
        }

        if(see(board, MoveTo(*curMove), board->squares[MoveTo(*curMove)], MoveFrom(*curMove), board->squares[MoveFrom(*curMove)]) < 0) {
            ++curMove;
            continue;
        }

        makeMove(board, *curMove, &undo);
    
        if(inCheck(board, !board->color)) {
            unmakeMove(board, *curMove, &undo);
            ++curMove;
            continue;
        }

        ++searchInfo->nodesCount;
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

    if(ABORT) {
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

void moveOrdering(Board* board, U16* moves, SearchInfo* searchInfo, int height, int depth) {
    U16* ptr = moves;
    U16 hashMove = tt[board->key & ttIndex].move;
    int i;

    for(i = 0; *ptr; ++i) {
        movePrice[height][i] = 0;
        U16 toPiece = pieceType(board->squares[MoveTo(*ptr)]);
        //int seeScore = see(board, MoveTo(*ptr), board->squares[MoveTo(*ptr)], MoveFrom(*ptr), board->squares[MoveFrom(*ptr)]);
        if(hashMove == *ptr) {
            movePrice[height][i] = 1000000000;
        } else if(toPiece) {
            U16 fromPiece = pieceType(board->squares[MoveFrom(*ptr)]);
            movePrice[height][i] = mvvLvaScores[fromPiece][toPiece] * 1000000;
            //movePrice[height][i] = 10000 * seeScore;
        } else if(searchInfo->killer[board->color][depth] == *ptr) {
            movePrice[height][i] = 100000;
        } else if(depth >= 2 && searchInfo->killer[board->color][depth-2] == *ptr) {
            movePrice[height][i] = 99999;
        } else if(searchInfo->secondKiller[board->color][depth] == *ptr) {
            movePrice[height][i] = 99998;
        } else if(depth >= 2 && searchInfo->secondKiller[board->color][depth-2] == *ptr) {
            movePrice[height][i] = 99997;
        } else if(!toPiece) {
            movePrice[height][i] = history[board->color][MoveFrom(*ptr)][MoveTo(*ptr)];
        }

        if(searchInfo->bestMove == *ptr && !height) {
            movePrice[height][i] = 1000000000;
        } 

        ++ptr;
    }

    sort(moves, i, height);
}

void sort(U16* moves, int count, int height) {
    int i, j, key;
    U16 keyMove;
    for (i = 1; i < count; i++)  { 
        key = movePrice[height][i];
        keyMove = moves[i];
        j = i - 1; 
    
        while (j >= 0 && movePrice[height][j] < key) { 
            movePrice[height][j + 1] = movePrice[height][j];
            moves[j + 1] = moves[j];
            --j;
        } 
        movePrice[height][j + 1] = key;
        moves[j + 1] = keyMove;
    }
}

void initSearch() {
    for(int attacker = 1; attacker < 7; ++attacker) {
        for(int victim = 1; victim < 7; ++victim) {
            mvvLvaScores[attacker][victim] = 64 * victim - attacker;
        }
    }

    for(int i = 0; i < MAX_PLY; ++i) {
        for(int j = 0; j < 64; ++j) {
            lmr[i][j]  = 0.75 + log(i) * log(j) / 2.25;
        }
    }

    clearHistory();
}

void resetSearchInfo(SearchInfo* info, TimeManager tm) {
    memset(info, 0, sizeof(SearchInfo));
    info->tm = tm;
    setAbort(0);
    compressHistory();
}

void replaceTransposition(Transposition* tr, Transposition new_tr, int height) {
    int score = new_tr.eval;

    if(score > MATE_SCORE - 100) {
        score += height;
    } else if(score < -MATE_SCORE + 100) {
        score -= height;
    }

    if(tr->age + 5 < ttAge || !tr->evalType) {
        replaceTranspositionEntry(tr, &new_tr);
        return;
    }

    if(new_tr.depth >= tr->depth) {
        new_tr.eval = score;
        if(new_tr.evalType == upperbound && tr->evalType != upperbound) {
            return;
        }
        replaceTranspositionEntry(tr, &new_tr);
    }
}

void setAbort(int val) {
    pthread_mutex_lock(&mutex);
    ABORT = val;
    pthread_mutex_unlock(&mutex);
}

void clearHistory() {
    memset(history, 0, 2*64*64 * sizeof(int));
}
void compressHistory() {
    for(int i = 0; i < 64; ++i) {
        for(int j = 0; j < 64; ++j) {
            history[WHITE][i][j] /= 100;
            history[BLACK][i][j] /= 100;
        }   
    }
}