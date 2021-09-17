#include "search.h"
#include "uci.h"
#include "types.h"

void* go(void* thread_data) {
    SearchArgs* args = (SearchArgs*)thread_data;
    iterativeDeeping(args->board, args->tm);
    return 0;
}

SearchInfo iterativeDeeping(Board* board, TimeManager tm) {
    ++ttAge;
    SearchInfo searchInfo;
    char bestMove[6];

    resetSearchInfo(&searchInfo, tm);
    startTimer(&searchInfo.timer);
    int eval = 0;
    for(int i = 1; i <= tm.depth; ++i) {
        eval = aspirationWindow(board, &searchInfo, i, eval);
        moveToString(searchInfo.bestMove, bestMove);
        if(ABORT && i > 1)
            break;
    }

    printf("info nodes %llu time %llu\n", searchInfo.nodesCount, getTime(&searchInfo.timer));
    SEARCH_COMPLETE = 1;
    __sync_synchronize();
    printf("bestmove %s\n", bestMove);
    fflush(stdout);
    
    return searchInfo;
}

int aspirationWindow(Board* board, SearchInfo* searchInfo, int depth, int score) {
    int delta = 15;
    int alpha = max(-MATE_SCORE, score - delta);
    int beta = min(MATE_SCORE, score + delta);

    if(depth <= 5)
        return search(board, searchInfo, -MATE_SCORE, MATE_SCORE, depth, 0);

    char bestMove[6];

    int f = score;
    
    while(abs(f) < MATE_SCORE - 1) {
        f = search(board, searchInfo, alpha, beta, depth, 0);
        
        moveToString(searchInfo->bestMove, bestMove);

        if(ABORT)
            break;

        int evalType = 0;

        if(f > alpha && f < beta)
            evalType = exact;

        if(f <= alpha) {
            beta = (alpha + beta) / 2;
            alpha = max(-MATE_SCORE, alpha - delta);
            evalType = upperbound;
        }

        if(f >= beta) {
            beta = min(MATE_SCORE, beta + delta);
            evalType = lowerbound;
        }

        printSearchInfo(searchInfo, board, depth, f, evalType);

        if(evalType == exact)
            break;

        delta += delta / 2;
    }

    return f;
}

int search(Board* board, SearchInfo* searchInfo, int alpha, int beta, int depth, int height) {
    searchInfo->selDepth = max(searchInfo->selDepth, height);
    ++searchInfo->nodesCount;
    
    if(ABORT)
        return 0;

    if(depth < 0 || depth > MAX_PLY - 1)
        depth = 0;

    //Mate Distance Pruning
    int mate_val = MATE_SCORE - height;
    if(mate_val < beta) {
        beta = mate_val;
        if(alpha >= mate_val)
            return mate_val;
    }

    mate_val = -MATE_SCORE + height;
    if(mate_val > alpha) {
        alpha = mate_val;
        if(beta <= mate_val)
            return mate_val;
    }

    int root = (height ? 0 : 1);
    int pvNode = (beta - alpha > 1);

    if((isDraw(board) && !root) || ABORT)
        return 0;

    int weInCheck = !!(inCheck(board, board->color));

    if(depth >= 3 && testAbort(getTime(&searchInfo->timer), searchInfo->nodesCount, &searchInfo->tm)) {
        setAbort(1);
        return 0;
    }

    U64 keyPosition = board->key;
    Transposition* ttEntry = &tt[keyPosition & ttIndex];

    int bestEntityIndex = getMaxDepthBucket(ttEntry, keyPosition);


    TranspositionEntity* bestEntity = NULL;
    if (bestEntityIndex != -1) {
        bestEntity = &ttEntry->entity[bestEntityIndex];
    }

    if (bestEntity) {
        int ttEval = evalFromTT(bestEntity->eval, height);
        // printf("%d\n", ttEntry->key == keyPosition);
        //TT analysis
        if (bestEntity->evalType && bestEntity->depth >= depth && !root) {
            if ((bestEntity->evalType == lowerbound && ttEval >= beta && !mateScore(bestEntity->eval)) ||
                (bestEntity->evalType == upperbound && ttEval <= alpha && !mateScore(bestEntity->eval)) ||
                bestEntity->evalType == exact) {
                return ttEval;
            }
        }
    }

    //go to quiescence search in leaf nodes
    if((depth <= 0 && !weInCheck) || height >= MAX_PLY - 1)
        return quiesceSearch(board, searchInfo, alpha, beta, height);

    //calculate static eval
    int staticEval = fullEval(board);

    //Null Move pruning
    
    int R = 2 + depth / 4;
    
    int pieceCount = popcount(board->colours[WHITE] | board->colours[BLACK]);
    if(NullMovePruningAllow && pieceCount > 7 && !pvNode && haveNoPawnMaterial(board) && !weInCheck && !root && !searchInfo->nullMoveSearch && depth > R && (staticEval >= beta || depth <= 4)) {
        makeNullMove(board);
        searchInfo->nullMoveSearch = 1;

        int eval = -search(board, searchInfo, -beta, -beta + 1, depth - 1 - R, height + 1);

        searchInfo->nullMoveSearch = 0;
        unmakeNullMove(board);

        if(eval >= beta)
            return beta;
    }

    //Reverse futility pruning
    if(!pvNode && !havePromotionPawn(board) && !weInCheck && depth <= 7 && staticEval - ReverseFutilityStep * depth > beta && ReverseFutilityPruningAllow)
        return staticEval;

    //Razoring
    if(!pvNode && !havePromotionPawn(board) && !weInCheck && depth <= 4 && staticEval + RazorMargin * depth < alpha && RazoringPruningAllow)
        return quiesceSearch(board, searchInfo, alpha, beta, height);

    movegen(board, moves[height]);
    moveOrdering(board, moves[height], searchInfo, height, depth);

    U16* curMove = moves[height];
    int movesCount = 0, pseudoMovesCount = 0, playedMovesCount = 0;
    Undo undo;

    int hashType = upperbound;
    U16 curBestMove = 0;

    searchInfo->killer[height + 1][0] = 0;
    searchInfo->killer[height + 1][1] = 0;

    while(*curMove) {
        int nextDepth = depth - 1;
        movePick(pseudoMovesCount, height);
        ++pseudoMovesCount;
        makeMove(board, *curMove, &undo);

        if(inCheck(board, !board->color)) {
            unmakeMove(board, *curMove, &undo);
            ++curMove;
            continue;
        }
        
        ++movesCount;

        int extensions = inCheck(board, board->color) || MovePromotionPiece(*curMove) == QUEEN;

        int quiteMove = (!undo.capturedPiece && MoveType(*curMove) != ENPASSANT_MOVE) && MoveType(*curMove) != PROMOTION_MOVE;

        if(root && depth > 12) {
            char moveStr[6];
            moveToString(*curMove, moveStr);
            printf("info currmove %s currmovenumber %d\n", moveStr, movesCount);
            fflush(stdout);
        }

        //Fulility pruning
        if(!pvNode && depth < 7 && !extensions && !root && FutilityPruningAllow) {
            if(staticEval + FutilityStep * depth + pVal(board, pieceType(undo.capturedPiece)) <= alpha) {
                unmakeMove(board, *curMove, &undo);
                ++curMove;
                continue;
            }
        }

        int reductions = lmr[min(depth, MAX_PLY-1)][min(playedMovesCount, 63)];
        ++playedMovesCount;

        int eval;
        if(movesCount == 1) {
            eval = -search(board, searchInfo, -beta, -alpha, nextDepth + extensions, height + 1);
        } else {
            if(LmrPruningAllow && playedMovesCount >= 3 && quiteMove) {
                eval = -search(board, searchInfo, -alpha - 1, -alpha, nextDepth + extensions - reductions, height + 1);
                if(eval > alpha)
                    eval = -search(board, searchInfo, -beta, -alpha, nextDepth + extensions, height + 1);
            } else {
                eval = -search(board, searchInfo, -alpha - 1, -alpha, nextDepth + extensions, height + 1);
    
                if(eval > alpha && eval < beta)
                    eval = -search(board, searchInfo, -beta, -alpha, nextDepth + extensions, height + 1);
            }
        }
        unmakeMove(board, *curMove, &undo);
        
        if(eval > alpha) {
            alpha = eval;
            curBestMove = *curMove;

            if(root && !ABORT)
                searchInfo->bestMove = *curMove;

            hashType = exact;
        }
        if(alpha >= beta) {
            hashType = lowerbound;

            if(!undo.capturedPiece) {
                if(searchInfo->killer[height][0])
                    searchInfo->killer[height][1] = searchInfo->killer[height][0];

                searchInfo->killer[height][0] = *curMove;
                history[board->color][MoveFrom(*curMove)][MoveTo(*curMove)] += (depth * depth);
            }

            break;
        }
        ++curMove;
    }

    if(ABORT)
        return 0;

    TranspositionEntity new_tt;
    new_tt.depth = depth;
    new_tt.age = ttAge;
    new_tt.evalType = hashType;
    new_tt.move = curBestMove;
    new_tt.key = keyPosition;
    new_tt.eval = evalToTT(alpha, height);

    // setTransposition(&new_tt, keyPosition, alpha, hashType, depth, curBestMove, ttAge, height);
    replaceTranspositionEntry(ttEntry, &new_tt, keyPosition);

    if(!movesCount) {
        if(inCheck(board, board->color))
            return -MATE_SCORE + height;
        else
            return 0;
    }

    return alpha;
}

int quiesceSearch(Board* board, SearchInfo* searchInfo, int alpha, int beta, int height) {
    searchInfo->selDepth = max(searchInfo->selDepth, height);

    U64 keyPosition = board->key;
    Transposition* ttEntry = &tt[keyPosition & ttIndex];

//    //TT analysis
//    int ttEval = evalFromTT(ttEntry->eval, height);
//    if(ttEntry->evalType && ttEntry->key == keyPosition && !TUNING_ENABLED) {
//        if((ttEntry->evalType == lowerbound && ttEval >= beta && !mateScore(ttEntry->eval)) ||
//           (ttEntry->evalType == upperbound && ttEval <= alpha && !mateScore(ttEntry->eval)) ||
//           ttEntry->evalType == exact) {
//               return ttEval;
//        }
//    }

    if(height >= MAX_PLY - 1)
        return fullEval(board);

    if(ABORT)
        return 0;
    
    int val = fullEval(board);
    if(val >= beta)
        return beta;

    int delta = QUEEN_EV_MG;
    if(havePromotionPawn(board))
        delta += (QUEEN_EV_MG - 200);

    if(val < alpha - delta)
        return val;

    if(alpha < val)
        alpha = val;

    attackgen(board, moves[height]);
    moveOrdering(board, moves[height], searchInfo, height, 0);
    U16* curMove = moves[height];
    Undo undo;
    int pseudoMovesCount = 0;
    while(*curMove) {
        if(ABORT)
            return 0;

        movePick(pseudoMovesCount, height);

        if(movePrice[height][pseudoMovesCount] < 0)
            break;

        ++pseudoMovesCount;

        makeMove(board, *curMove, &undo);
    
        if(inCheck(board, !board->color)) {
            unmakeMove(board, *curMove, &undo);
            ++curMove;
            continue;
        }

        ++searchInfo->nodesCount;
        int score = -quiesceSearch(board, searchInfo, -beta, -alpha, height + 1);

        unmakeMove(board, *curMove, &undo);
        if(score >= beta)
            return beta;
        if(score > alpha)
           alpha = score;

        ++curMove;
    }

    if(ABORT)
        return 0;

    return alpha;
}

U64 perftTest(Board* board, int depth, int height) {
    if(!depth)
        return 1;


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
                for(int i = 0; i < height; ++i)
                    printf(" ");

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

        if(!(end - start))
            end = start + 1;
        
        printf("Perft %d: %llu; speed: %llu; time: %.3fs\n", i, nodes, nodes / (end - start), (end - start) / 1000.);
    }
}

void moveOrdering(Board* board, U16* mvs, SearchInfo* searchInfo, int height, int depth) {
    if(depth > MAX_PLY - 1)
        depth = MAX_PLY - 1;

    U16* ptr = mvs;
    // U16 hashMove = tt[board->key & ttIndex].move;
    Transposition* ttEntry = &tt[board->key & ttIndex];
    int i;

    for(i = 0; *ptr; ++i) {
        int isHashMove = 0;
        movePrice[height][i] = 0;
        U16 toPiece = pieceType(board->squares[MoveTo(*ptr)]);

        for (int j = 0; j < BUCKETS_N; ++j) {
            if (*ptr == ttEntry->entity[j].move) {
                movePrice[height][i] = 1000000000 + ttEntry->entity[j].depth;
                isHashMove = 1;
                break;
            }
        }

        if (isHashMove) {
            ++ptr;
            continue;
        }
        
        if(toPiece)
            movePrice[height][i] = mvvLvaScores[pieceType(board->squares[MoveFrom(*ptr)])][toPiece] * 1000000;
        else if(depth < MAX_PLY && searchInfo->killer[height][0] == *ptr)
            movePrice[height][i] = 100000;
        else if(depth >= 2 && depth < MAX_PLY && searchInfo->killer[height - 2][0] == *ptr)
            movePrice[height][i] = 99999;
        else if(depth < MAX_PLY && searchInfo->killer[height][1] == *ptr)
            movePrice[height][i] = 99998;
        else if(depth >= 2 && depth < MAX_PLY && searchInfo->killer[height - 2][1] == *ptr)
            movePrice[height][i] = 99997;
        else if(!toPiece)
            movePrice[height][i] = history[board->color][MoveFrom(*ptr)][MoveTo(*ptr)];

        if(MoveType(*ptr) == ENPASSANT_MOVE)
            movePrice[height][i] = mvvLvaScores[PAWN][PAWN] * 1000000;

        if(toPiece) {
            int seeScore = see(board, MoveTo(*ptr), board->squares[MoveTo(*ptr)], MoveFrom(*ptr), board->squares[MoveFrom(*ptr)]);
            if(seeScore < 0 /*&& hashMove != *ptr*/)
                movePrice[height][i] = seeScore;
        }

        
        if(MoveType(*ptr) == PROMOTION_MOVE) {
            if(MovePromotionPiece(*ptr) == QUEEN)
                movePrice[height][i] = 999999999;
            else
                movePrice[height][i] = 0;
        } 
        
        if(searchInfo->bestMove == *ptr && !height)
            movePrice[height][i] = 1000000000;

        ++ptr;
    }
}

void movePick(int moveNumber, int height) {
    int bestPrice = movePrice[height][moveNumber];
    int bestNumber = moveNumber;

    for(int i = moveNumber + 1; moves[height][i]; ++i) {
        if(movePrice[height][i] > bestPrice) {
            bestNumber = i;
            bestPrice = movePrice[height][i];
        }
    }

    U16 tmpMove = moves[height][moveNumber];
    moves[height][moveNumber] = moves[height][bestNumber];
    moves[height][bestNumber] = tmpMove;

    int tmpPrice = movePrice[height][moveNumber];
    movePrice[height][moveNumber] = movePrice[height][bestNumber];
    movePrice[height][bestNumber] = tmpPrice;
}

void initSearch() {
    for(int attacker = 1; attacker < 7; ++attacker) {
        for(int victim = 1; victim < 7; ++victim)
            mvvLvaScores[attacker][victim] = 64 * victim - attacker;
    }

    for(int i = 1; i < MAX_PLY; ++i) {
        for(int j = 1; j < 64; ++j) {
            lmr[i][j] = 0.75 + log(i) * log(j) / 2.25;
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