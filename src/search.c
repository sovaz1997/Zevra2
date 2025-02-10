#include "search.h"
#include "uci.h"
#include "types.h"

void *go(void *thread_data) {
    SearchArgs *args = (SearchArgs *) thread_data;
    iterativeDeeping(args->board, args->tm);
    return 0;
}

SearchInfo iterativeDeeping(Board *board, TimeManager tm) {
    ++ttAge;
    SearchInfo searchInfo;
    char bestMove[6];

    resetSearchInfo(&searchInfo, tm);
    startTimer(&searchInfo.timer);
    int eval = 0;
    int prevEval = 0;
    for (int i = 1; i <= tm.depth; ++i) {
        prevEval = eval;
        eval = aspirationWindow(board, &searchInfo, i, eval);

        moveToString(searchInfo.bestMove, bestMove);
        if (ABORT && i > 1)
            break;
    }

    printf("info nodes %llu time %llu\n", searchInfo.nodesCount, getTime(&searchInfo.timer));
    SEARCH_COMPLETE = 1;
    __sync_synchronize();
    printf("bestmove %s\n", bestMove);
    fflush(stdout);

    searchInfo.eval = prevEval;

    return searchInfo;
}

int aspirationWindow(Board *board, SearchInfo *searchInfo, int depth, int score) {
    int delta = 15;
    int alpha = max(-MATE_SCORE, score - delta);
    int beta = min(MATE_SCORE, score + delta);

    if (depth <= 5)
        return search(board, searchInfo, -MATE_SCORE, MATE_SCORE, depth, 0);

    char bestMove[6];

    int f = score;

    while (abs(f) < MATE_SCORE - 1) {
        f = search(board, searchInfo, alpha, beta, depth, 0);

        moveToString(searchInfo->bestMove, bestMove);

        if (ABORT)
            break;

        int evalType = 0;

        if (f > alpha && f < beta)
            evalType = exact;

        if (f <= alpha) {
            beta = (alpha + beta) / 2;
            alpha = max(-MATE_SCORE, alpha - delta);
            evalType = upperbound;
        }

        if (f >= beta) {
            beta = min(MATE_SCORE, beta + delta);
            evalType = lowerbound;
        }

        printSearchInfo(searchInfo, board, depth, f, evalType);

        if (evalType == exact)
            break;

        delta += delta / 2;
    }

    return f;
}

int search(Board *board, SearchInfo *searchInfo, int alpha, int beta, int depth, int height) {
    searchInfo->selDepth = max(searchInfo->selDepth, height);
    ++searchInfo->nodesCount;

    if (ABORT)
        return 0;

    if (depth < 0 || depth > MAX_PLY - 1)
        depth = 0;

    //Mate Distance Pruning
    int mate_val = MATE_SCORE - height;
    if (mate_val < beta) {
        beta = mate_val;
        if (alpha >= mate_val)
            return mate_val;
    }

    mate_val = -MATE_SCORE + height;
    if (mate_val > alpha) {
        alpha = mate_val;
        if (beta <= mate_val)
            return mate_val;
    }

    int root = (height ? 0 : 1);
    int pvNode = (beta - alpha > 1);

    if (root) {
      for (int i = 0; i < 256; i++) {
        temperatureOffsets[i] = temperature == 0 ? 0 : rand() % temperature;
      }
    }

    if ((isDraw(board) && !root) || ABORT)
        return 0;

    if (depth >= 3 && testAbort(getTime(&searchInfo->timer), searchInfo->nodesCount, &searchInfo->tm)) {
        setAbort(1);
        return 0;
    }

    U64 keyPosition = board->key;
    Transposition *ttEntry = getTTEntry(keyPosition);

    if (!pvNode && ttEntry && ttEntry->key == board->key && ttEntry->evalType && ttEntry->depth >= depth && !root) {
        int ttEval = evalFromTT(ttEntry->eval, height);

        //TT analysis
        if ((ttEntry->evalType == lowerbound && ttEval >= beta && !mateScore(ttEntry->eval)) ||
            (ttEntry->evalType == upperbound && ttEval <= alpha && !mateScore(ttEntry->eval)) ||
            ttEntry->evalType == exact) {
            return ttEval;
        }
    }

    int weInCheck = !!(inCheck(board, board->color));

    //go to quiescence search in leaf nodes
    if ((depth <= 0 && !weInCheck) || height >= MAX_PLY - 1)
        return quiesceSearch(board, searchInfo, alpha, beta, height);

    //calculate static eval
    int staticEval = fullEval(board);


    //Null Move pruning
	#if ENABLE_NULL_MOVE_PRUNING
    int R = 2 + depth / 4;
    int pieceCount = popcount(board->colours[WHITE] | board->colours[BLACK]);
    if (!pvNode && pieceCount > 7 && !weInCheck && !root && haveNoPawnMaterial(board) &&
        !searchInfo->nullMoveSearch && depth > R && (staticEval >= beta || depth <= 4)) {
        makeNullMove(board);
        searchInfo->nullMoveSearch = 1;

        int eval = -search(board, searchInfo, -beta, -beta + 1, depth - 1 - R, height + 1);

        searchInfo->nullMoveSearch = 0;
        unmakeNullMove(board);

        if (eval >= beta)
            return beta;
    }
    #endif


    if (!pvNode && !weInCheck && !havePromotionPawn(board)) {
        //Reverse futility pruning
        #if ENABLE_RAZORING
        if (depth <= 7 &&
            staticEval - ReverseFutilityStep * depth > beta)
        	return staticEval;
        #endif

    	//Razoring
        #if ENABLE_REVERSE_FUTILITY_PRUNING
    	if (depth <= 7 && staticEval + RazorMargin * depth < alpha)
        	return quiesceSearch(board, searchInfo, alpha, beta, height);
        #endif
    }

    movegen(board, moves[height]);
    moveOrdering(board, moves[height], searchInfo, height, depth);

    U16 *curMove = moves[height];
    int movesCount = 0, pseudoMovesCount = 0, playedMovesCount = 0;
    Undo undo;

    int hashType = upperbound;
    U16 curBestMove = 0;

    searchInfo->killer[height + 1][0] = 0;
    searchInfo->killer[height + 1][1] = 0;

    while (*curMove) {
        int bonus = mateScore(alpha) || !root ? 0 : temperatureOffsets[pseudoMovesCount];

        int nextDepth = depth - 1;
        movePick(pseudoMovesCount, height);
        ++pseudoMovesCount;
        makeMove(board, *curMove, &undo);


        if (inCheck(board, !board->color)) {
            unmakeMove(board, *curMove, &undo);
            ++curMove;
            continue;
        }

        ++movesCount;

        int extensions = inCheck(board, board->color) || MovePromotionPiece(*curMove) == QUEEN;

        int quiteMove =
                (!undo.capturedPiece && MoveType(*curMove) != ENPASSANT_MOVE) && MoveType(*curMove) != PROMOTION_MOVE;

        if (root && depth > 12) {
            char moveStr[6];
            moveToString(*curMove, moveStr);
            if (!SHOULD_HIDE_SEARCH_INFO_LOGS) {
                printf("info currmove %s currmovenumber %d\n", moveStr, movesCount);
                fflush(stdout);
            }
        }

        //Fulility pruning
        #if ENABLE_FUTILITY_PRUNING
        if (!pvNode && depth < 7 && !extensions && !root) {
            if (staticEval + FutilityStep * depth + pVal(board, pieceType(undo.capturedPiece)) <= alpha) {
                unmakeMove(board, *curMove, &undo);
                ++curMove;
                continue;
            }
        }
        #endif

        int reductions = lmr[min(depth, MAX_PLY - 1)][min(playedMovesCount, 63)];
        ++playedMovesCount;

        int eval;
        if (movesCount == 1) {
            eval = -search(board, searchInfo, -beta + bonus, -alpha + bonus, nextDepth + extensions, height + 1) - bonus;
        } else {
            if (LmrPruningAllow && playedMovesCount >= 3 && quiteMove) {
                eval = -search(board, searchInfo, -alpha - 1 + bonus, -alpha + bonus, nextDepth + extensions - reductions, height + 1) - bonus;
                if (eval > alpha)
                    eval = -search(board, searchInfo, -beta + bonus, -alpha + bonus, nextDepth + extensions, height + 1) - bonus;
            } else {
                eval = -search(board, searchInfo, -alpha - 1 + bonus, -alpha + bonus, nextDepth + extensions, height + 1) - bonus;

                if (eval > alpha && eval < beta)
                    eval = -search(board, searchInfo, -beta + bonus, -alpha + bonus, nextDepth + extensions, height + 1) - bonus;
            }
        }
        unmakeMove(board, *curMove, &undo);

        if (eval > alpha) {
            alpha = eval;
            curBestMove = *curMove;

            if (root && !ABORT)
                searchInfo->bestMove = *curMove;

            hashType = exact;
        }
        if (alpha >= beta) {
            hashType = lowerbound;

            if (!undo.capturedPiece) {
                if (searchInfo->killer[height][0])
                    searchInfo->killer[height][1] = searchInfo->killer[height][0];

                searchInfo->killer[height][0] = *curMove;
                history[board->color][MoveFrom(*curMove)][MoveTo(*curMove)] += (depth * depth);
            }

            break;
        }
        ++curMove;
    }

    if (ABORT)
        return 0;

    Transposition new_tt;
    new_tt.depth = depth;
    new_tt.age = ttAge;
    new_tt.evalType = hashType;
    new_tt.move = curBestMove;
    new_tt.key = keyPosition;
    new_tt.eval = evalToTT(alpha, height);

    replaceTranspositionEntry(&new_tt, keyPosition);

    if (!movesCount) {
        if (inCheck(board, board->color))
            return -MATE_SCORE + height;
        else
            return 0;
    }

    return alpha;
}

int quiesceSearch(Board *board, SearchInfo *searchInfo, int alpha, int beta, int height) {
    searchInfo->selDepth = max(searchInfo->selDepth, height);

    U64 keyPosition = board->key;
    Transposition *ttEntry = getTTEntry(keyPosition);

    if (ttEntry && ttEntry->key == board->key) {
        int ttEval = evalFromTT(ttEntry->eval, height);
        if (ttEntry->evalType) {
            if ((ttEntry->evalType == lowerbound && ttEval >= beta && !mateScore(ttEntry->eval)) ||
                (ttEntry->evalType == upperbound && ttEval <= alpha && !mateScore(ttEntry->eval)) ||
                ttEntry->evalType == exact) {
                return ttEval;
            }
        }
    }

    if (height >= MAX_PLY - 1)
        return fullEval(board);

    if (ABORT)
        return 0;

    int val = fullEval(board);
    if (val >= beta)
        return beta;

    int delta = QUEEN_EV_MG;
    if (havePromotionPawn(board))
        delta += (QUEEN_EV_MG - 200);

    if (val < alpha - delta)
        return val;

    if (alpha < val)
        alpha = val;

    attackgen(board, moves[height]);
    moveOrdering(board, moves[height], searchInfo, height, 0);
    U16 *curMove = moves[height];
    Undo undo;
    int pseudoMovesCount = 0;
    while (*curMove) {
        if (ABORT)
            return 0;

        movePick(pseudoMovesCount, height);

        if (movePrice[height][pseudoMovesCount] < 0)
            break;

        ++pseudoMovesCount;

        makeMove(board, *curMove, &undo);

        if (inCheck(board, !board->color)) {
            unmakeMove(board, *curMove, &undo);
            ++curMove;
            continue;
        }

        ++searchInfo->nodesCount;
        int score = -quiesceSearch(board, searchInfo, -beta, -alpha, height + 1);

        unmakeMove(board, *curMove, &undo);
        if (score >= beta)
            return beta;
        if (score > alpha)
            alpha = score;

        ++curMove;
    }

    if (ABORT)
        return 0;

    return alpha;
}

U64 perftTest(Board *board, int depth, int height) {
    if (!depth)
        return 1;


    movegen(board, moves[height]);

    U64 result = 0;
    U16 *curMove = moves[height];
    Undo undo;
    while (*curMove) {
        makeMove(board, *curMove, &undo);

        U64 count = 0;
        if (!inCheck(board, !board->color)) {
            count = perftTest(board, depth - 1, height + 1);

            if (!height) {
                char mv[6];
                moveToString(*curMove, mv);
                for (int i = 0; i < height; ++i)
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

void perft(Board *board, int depth) {
    for (int i = 1; i <= depth; ++i) {
        clock_t start = clock();
        U64 nodes = perftTest(board, i, 0);
        clock_t end = clock();
        double speed = (double)nodes / ((double)end - (double)start);

        if (!(end - start))
            end = start + 1;

        printf("Perft %d: %llu; speed: %.1fMnps; time: %.3fs\n", i, nodes, speed, (end - start) / 1000000.);
    }
}

void moveOrdering(Board *board, U16 *mvs, SearchInfo *searchInfo, int height, int depth) {
    if (depth > MAX_PLY - 1)
        depth = MAX_PLY - 1;

    U16 *ptr = mvs;
    Transposition *ttEntry = getTTEntry(board->key);
    int i;

    for (i = 0; *ptr; ++i) {
        int isHashMove = 0;
        movePrice[height][i] = 0;
        U16 toPiece = pieceType(board->squares[MoveTo(*ptr)]);

        if (*ptr == ttEntry->move && ttEntry->key == board->key) {
            movePrice[height][i] = 1000000000000000llu + ttEntry->depth;
            isHashMove = 1;
        }

        if (isHashMove) {
            ++ptr;
            continue;
        }

        if (toPiece)
            movePrice[height][i] = mvvLvaScores[pieceType(board->squares[MoveFrom(*ptr)])][toPiece] * 1000000000000llu;
        else if (depth < MAX_PLY && searchInfo->killer[height][0] == *ptr)
            movePrice[height][i] = 100000000000llu;
        else if (depth >= 2 && depth < MAX_PLY && searchInfo->killer[height - 2][0] == *ptr)
            movePrice[height][i] = 99999000000llu;
        else if (depth < MAX_PLY && searchInfo->killer[height][1] == *ptr)
            movePrice[height][i] = 99998000000llu;
        else if (depth >= 2 && depth < MAX_PLY && searchInfo->killer[height - 2][1] == *ptr)
            movePrice[height][i] = 99997000000llu;
        else {
            movePrice[height][i] = history[board->color][MoveFrom(*ptr)][MoveTo(*ptr)];
        }

        if (MoveType(*ptr) == ENPASSANT_MOVE)
            movePrice[height][i] = mvvLvaScores[PAWN][PAWN] * 1000000000000llu;

        if (toPiece) {
            int seeScore = see(board, MoveTo(*ptr), board->squares[MoveTo(*ptr)], MoveFrom(*ptr),
                               board->squares[MoveFrom(*ptr)]);

            if (seeScore < 0) {
                movePrice[height][i] = seeScore;
            }
        }


        if (MoveType(*ptr) == PROMOTION_MOVE) {
            if (MovePromotionPiece(*ptr) == QUEEN) {
                movePrice[height][i] = 999999999000000llu;
            } else {
                movePrice[height][i] = 0;
            }
        }

        if (searchInfo->bestMove == *ptr && !height) {
            movePrice[height][i] = 10000000000000000llu;
        }

        ++ptr;
    }
}

void movePick(int moveNumber, int height) {
    long long bestPrice = movePrice[height][moveNumber];
    int bestNumber = moveNumber;

    for (int i = moveNumber + 1; moves[height][i]; ++i) {
        if (movePrice[height][i] > bestPrice) {
            bestNumber = i;
            bestPrice = movePrice[height][i];
        }
    }

    U16 tmpMove = moves[height][moveNumber];
    moves[height][moveNumber] = moves[height][bestNumber];
    moves[height][bestNumber] = tmpMove;

    long long tmpPrice = movePrice[height][moveNumber];
    movePrice[height][moveNumber] = movePrice[height][bestNumber];
    movePrice[height][bestNumber] = tmpPrice;
}

void initSearch() {
    for (int attacker = 1; attacker < 7; ++attacker) {
        for (int victim = 1; victim < 7; ++victim)
            mvvLvaScores[attacker][victim] = 64 * victim - attacker;
    }

    for (int i = 1; i < MAX_PLY; ++i) {
        for (int j = 1; j < 64; ++j) {
            lmr[i][j] = 0.75 + log(i) * log(j) / 2.25;
        }
    }

    clearHistory();
}

void resetSearchInfo(SearchInfo *info, TimeManager tm) {
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
    memset(history, 0, 2 * 64 * 64 * sizeof(int));
}

void compressHistory() {
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            history[WHITE][i][j] /= 100;
            history[BLACK][i][j] /= 100;
        }
    }
}