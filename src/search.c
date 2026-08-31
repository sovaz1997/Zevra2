#include "search.h"
#include "uci.h"
#include "types.h"

// Definitions for globals declared extern in search.h
volatile int ABORT;
volatile int SEARCH_COMPLETE;

long long history[2][64][64];
long long contHist[14][64][14][64];
long long contHist2[14][64][14][64];
U8 contPiece[MAX_PLY + 2];
U8 contTo[MAX_PLY + 2];

int FutilityStep;
int ReverseFutilityStep;
int RazorMargin;
int LmpMaxDepth;
int LmpBase;
int NmpBase;
int NmpDiv;
int AspirationDelta;
int LmrBaseX100;
int LmrDivX100;
int LmrHistoryDiv;
int LmpPruningAllow;

U16 moves[MAX_PLY][256];
int improving[256];
U8 temperatureOffsets[256];
long long movePrice[MAX_PLY][256];
int mvvLvaScores[7][7];
int lmr[MAX_PLY][64];
int lmpTable[2][MAX_PLY];

SearchStats g_stats;

void resetSearchStats() {
    memset(&g_stats, 0, sizeof(g_stats));
}

static double pct(U64 num, U64 den) {
    return den ? 100.0 * (double) num / (double) den : 0.0;
}

// Gravity history update: pulls *h toward +/-MAX_HISTORY, so recent signal dominates
// and values self-bound (no periodic aging needed).
static void histBonus(long long *h, int bonus) {
    if (bonus > MAX_HISTORY) bonus = MAX_HISTORY;
    if (bonus < -MAX_HISTORY) bonus = -MAX_HISTORY;
    int mag = bonus < 0 ? -bonus : bonus;
    *h += bonus - (*h) * mag / MAX_HISTORY;
}

// Apply bonus/malus to a quiet move across main + 1-ply + 2-ply history tables.
static void updateQuietHist(int color, U16 move, U8 piece, int height, int bonus) {
    int from = MoveFrom(move), to = MoveTo(move);
    histBonus(&history[color][from][to], bonus);
    if (height >= 1) histBonus(&contHist[contPiece[height - 1]][contTo[height - 1]][piece][to], bonus);
    if (height >= 2) histBonus(&contHist2[contPiece[height - 2]][contTo[height - 2]][piece][to], bonus);
}

void printSearchStats() {
#ifdef SEARCH_STATS
    printf("---- search stats ----\n");
    printf("main nodes            : %llu\n", (unsigned long long) g_stats.nodes);
    printf("fail-high first-move  : %.1f%%  (ordering quality; aim >90%%)\n",
           pct(g_stats.failHighFirst, g_stats.failHigh));
    printf("null-move cut rate    : %.1f%%  (%llu/%llu attempts)\n",
           pct(g_stats.nmpCut, g_stats.nmpTries),
           (unsigned long long) g_stats.nmpCut, (unsigned long long) g_stats.nmpTries);
    printf("LMR re-search rate    : %.1f%%  (>~25%% = reducing too hard, <~5%% = too soft)\n",
           pct(g_stats.lmrReSearch, g_stats.lmrTries));
    printf("aspiration fail rate  : %.1f%%  (%llu/%llu windows)\n",
           pct(g_stats.aspFail, g_stats.aspTries),
           (unsigned long long) g_stats.aspFail, (unsigned long long) g_stats.aspTries);
    printf("node-returning prunes (%% of nodes):\n");
    printf("  RFP                 : %.2f%%  (%llu)\n", pct(g_stats.rfpCut, g_stats.nodes), (unsigned long long) g_stats.rfpCut);
    printf("  razoring            : %.2f%%  (%llu)\n", pct(g_stats.razorCut, g_stats.nodes), (unsigned long long) g_stats.razorCut);
    printf("  null-move cutoffs   : %.2f%%  (%llu)\n", pct(g_stats.nmpCut, g_stats.nodes), (unsigned long long) g_stats.nmpCut);
    printf("futility prunes/node  : %.2f   (%llu moves pruned; per-move, not per-node)\n",
           g_stats.nodes ? (double) g_stats.futilityCut / (double) g_stats.nodes : 0.0,
           (unsigned long long) g_stats.futilityCut);
    printf("LMP prunes/node       : %.2f   (%llu quiet moves pruned; per-move, not per-node)\n",
           g_stats.nodes ? (double) g_stats.lmpCut / (double) g_stats.nodes : 0.0,
           (unsigned long long) g_stats.lmpCut);
    printf("----------------------\n");
    fflush(stdout);
#endif
#ifdef VERIFY_LMR
    printf("LMR over-reduction    : %.1f%%  (%llu/%llu reduced+skipped moves would beat alpha at full depth)\n",
           pct(g_stats.lmrVerMissed, g_stats.lmrVerChecked),
           (unsigned long long) g_stats.lmrVerMissed, (unsigned long long) g_stats.lmrVerChecked);
    fflush(stdout);
#endif
#if !defined(SEARCH_STATS) && !defined(VERIFY_LMR)
    (void) pct;
#endif
}

void *go(void *thread_data) {
    SearchArgs *args = (SearchArgs *) thread_data;
    iterativeDeeping(args->board, args->tm);
    free(args);   // allocated in uci.c before pthread_create
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
    U16 lastBest = 0;
    int stability = 0;
    for (int i = 1; i <= tm.depth; ++i) {
        prevEval = eval;
        eval = aspirationWindow(board, &searchInfo, i, eval);

        moveToString(searchInfo.bestMove, bestMove);
        if (ABORT && i > 1)
            break;

        // best-move stability tracking (drives the adaptive time correction)
        if (searchInfo.bestMove == lastBest) stability++;
        else stability = 0;
        lastBest = searchInfo.bestMove;

        // stop starting new depths once the soft (stability-adjusted) limit is hit
        if (shouldStopDeepening(&tm, getTime(&searchInfo.timer), i, stability, eval, prevEval))
            break;
    }

    if (!SHOULD_HIDE_SEARCH_INFO_LOGS)
        printf("info nodes %llu time %llu\n", searchInfo.nodesCount, getTime(&searchInfo.timer));
    SEARCH_COMPLETE = 1;
    __sync_synchronize();
    if (!SHOULD_HIDE_SEARCH_INFO_LOGS) {
        printf("bestmove %s\n", bestMove);
        fflush(stdout);
    }

    searchInfo.eval = prevEval;

    return searchInfo;
}

int aspirationWindow(Board *board, SearchInfo *searchInfo, int depth, int score) {
    int delta = AspirationDelta;
    int alpha = max(-MATE_SCORE, score - delta);
    int beta = min(MATE_SCORE, score + delta);

    if (depth <= 5)
        return search(board, searchInfo, -MATE_SCORE, MATE_SCORE, depth, 0);

    char bestMove[6];

    int f = score;

    while (abs(f) < MATE_SCORE - 1) {
        f = search(board, searchInfo, alpha, beta, depth, 0);
        STAT(aspTries);

        moveToString(searchInfo->bestMove, bestMove);

        if (ABORT)
            break;

        int evalType = 0;

        if (f > alpha && f < beta)
            evalType = exact;

        if (f <= alpha) {
            STAT(aspFail);
            beta = (alpha + beta) / 2;
            alpha = max(-MATE_SCORE, alpha - delta);
            evalType = upperbound;
        }

        if (f >= beta) {
            STAT(aspFail);
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
    STAT(nodes);

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

    improving[height] = staticEval;

    int hasImproving = 0;

    if (!weInCheck && !searchInfo->nullMoveSearch) {
    	if (height >= 4) {
        	hasImproving = improving[height] > improving[height - 4];
    	} else if (height >= 2) {
            hasImproving = improving[height] > improving[height - 2];
        } else {
            hasImproving = 0;
        }
    }

    //Null Move pruning
	#if ENABLE_NULL_MOVE_PRUNING
    int R = NmpBase + depth / NmpDiv;
    int pieceCount = popcount(board->colours[WHITE] | board->colours[BLACK]);
    if (!pvNode && pieceCount > 7 && !weInCheck && !root && haveNoPawnMaterial(board) &&
        !searchInfo->nullMoveSearch && depth > R && staticEval >= beta) {
        STAT(nmpTries);
        makeNullMove(board);
        searchInfo->nullMoveSearch = 1;
        contPiece[height] = 0; contTo[height] = 0;   // child sees "no previous move"

        int eval = -search(board, searchInfo, -beta, -beta + 1, depth - 1 - R, height + 1);

        searchInfo->nullMoveSearch = 0;
        unmakeNullMove(board);

        if (eval >= beta) {
            STAT(nmpCut);
            return beta;
        }
    }
    #endif


    if (!pvNode && !weInCheck && !havePromotionPawn(board)) {
        //Reverse futility pruning
        #if ENABLE_REVERSE_FUTILITY_PRUNING
        if (depth <= 7 &&
            staticEval - ReverseFutilityStep  * depth > beta) {
        	STAT(rfpCut);
        	return staticEval;
        }
        #endif

    	//Razoring
        #if ENABLE_RAZORING
    	if (depth <= 7 && staticEval + RazorMargin * depth < alpha) {
        	STAT(razorCut);
        	return quiesceSearch(board, searchInfo, alpha, beta, height);
        }
        #endif
    }

    movegen(board, moves[height]);
    moveOrdering(board, moves[height], searchInfo, height, depth);

    U16 *curMove = moves[height];
    int movesCount = 0, pseudoMovesCount = 0, playedMovesCount = 0;
    Undo undo;

    int hashType = upperbound;
    U16 curBestMove = 0;

    U16 quietMoves[64]; U8 quietPieces[64]; int nQuiets = 0;   // searched quiets (for malus)

    searchInfo->killer[height + 1][0] = 0;
    searchInfo->killer[height + 1][1] = 0;

    while (*curMove) {
        int bonus = mateScore(alpha) || !root ? 0 : temperatureOffsets[pseudoMovesCount];

        int nextDepth = depth - 1;
        movePick(pseudoMovesCount, height);
        ++pseudoMovesCount;
        U8 movedPiece = board->squares[MoveFrom(*curMove)];   // before makeMove
        makeMove(board, *curMove, &undo);


        if (inCheck(board, !board->color)) {
            unmakeMove(board, *curMove, &undo);
            ++curMove;
            continue;
        }

        ++movesCount;
        // record this move so child nodes can read it as their "previous move"
        contPiece[height] = movedPiece;
        contTo[height] = MoveTo(*curMove);

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

        //Late Move Pruning: at shallow depth in non-PV nodes, once enough moves
        //have been tried, skip the remaining quiet ones -- they are ordered last
        //and are very unlikely to raise alpha. Trades accuracy for depth.
        #if ENABLE_LATE_MOVE_PRUNING
        if (LmpPruningAllow && !pvNode && !root && !weInCheck && quiteMove && !extensions &&
            depth <= LmpMaxDepth && !mateScore(alpha) &&
            movesCount > lmpTable[hasImproving][depth]) {
            STAT(lmpCut);
            unmakeMove(board, *curMove, &undo);
            ++curMove;
            continue;
        }
        #endif

        //Fulility pruning
        #if ENABLE_FUTILITY_PRUNING
        if (!pvNode && depth < 7 && !extensions && !root) {
            if (staticEval + FutilityStep * depth + pVal(board, pieceType(undo.capturedPiece)) <= alpha) {
                STAT(futilityCut);
                unmakeMove(board, *curMove, &undo);
                ++curMove;
                continue;
            }
        }
        #endif

        if (quiteMove && nQuiets < 64) {   // remember searched quiets for malus on cutoff
            quietMoves[nQuiets] = *curMove;
            quietPieces[nQuiets] = movedPiece;
            ++nQuiets;
        }

        int reductions = lmr[min(depth, MAX_PLY - 1)][min(playedMovesCount, 63)];
        ++playedMovesCount;

        // Selective LMR: adjust the reduction by context so we reduce junk harder but
        // spare important quiets (this is the lever SPSA can push the base against).
        if (reductions > 0) {
            reductions -= pvNode;                    // reduce PV lines less
            if (!hasImproving) reductions += 1;      // reduce more when not improving
            if (*curMove == searchInfo->killer[height][0] ||
                *curMove == searchInfo->killer[height][1])
                reductions -= 1;                     // killers are important quiets
            {   // history-based: good history -> reduce less, bad -> more
                int mColor = !board->color;          // board is in the made state here
                U16 hto = MoveTo(*curMove);
                long long h = history[mColor][MoveFrom(*curMove)][hto];
                if (height >= 1) h += contHist[contPiece[height - 1]][contTo[height - 1]][movedPiece][hto];
                if (height >= 2) h += contHist2[contPiece[height - 2]][contTo[height - 2]][movedPiece][hto];
                int hr = h / LmrHistoryDiv;
                if (hr > 2) hr = 2; else if (hr < -2) hr = -2;
                reductions -= hr;
            }
            if (reductions < 0) reductions = 0;
        }

        int eval;
        if (movesCount == 1) {
            eval = -search(board, searchInfo, -beta + bonus, -alpha + bonus, nextDepth + extensions, height + 1) - bonus;
        } else {
            if (LmrPruningAllow && playedMovesCount >= 3 && quiteMove) {
                if (reductions > 0)
                    STAT(lmrTries);
                eval = -search(board, searchInfo, -alpha - 1 + bonus, -alpha + bonus, nextDepth + extensions - reductions, height + 1) - bonus;
#ifdef VERIFY_LMR
                // Ground-truth check: this move was reduced and (about to be)
                // skipped without a re-search. Would a full-depth search beat
                // alpha? If so the reduction blindly hid a real move.
                if (reductions > 0 && eval <= alpha) {
                    g_stats.lmrVerChecked++;
                    int full = -search(board, searchInfo, -alpha - 1 + bonus, -alpha + bonus, nextDepth + extensions, height + 1) - bonus;
                    if (full > alpha)
                        g_stats.lmrVerMissed++;
                }
#endif
                if (eval > alpha) {
                    if (reductions > 0)
                        STAT(lmrReSearch);
                    eval = -search(board, searchInfo, -beta + bonus, -alpha + bonus, nextDepth + extensions, height + 1) - bonus;
                }
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
            STAT(failHigh);
            if (movesCount == 1)
                STAT(failHighFirst);

            int histBns = depth * depth;
            if (!undo.capturedPiece) {
                if (searchInfo->killer[height][0])
                    searchInfo->killer[height][1] = searchInfo->killer[height][0];

                searchInfo->killer[height][0] = *curMove;
                updateQuietHist(board->color, *curMove, movedPiece, height, histBns);
            }
            // malus to quiet moves that were searched first but failed to cut
            for (int q = 0; q < nQuiets; ++q) {
                if (quietMoves[q] == *curMove) continue;
                updateQuietHist(board->color, quietMoves[q], quietPieces[q], height, -histBns);
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

    // previous move(s) into this node, for continuation-history lookup
    U8 pp1 = height >= 1 ? contPiece[height - 1] : 0;
    U8 pt1 = height >= 1 ? contTo[height - 1] : 0;
    U8 pp2 = height >= 2 ? contPiece[height - 2] : 0;
    U8 pt2 = height >= 2 ? contTo[height - 2] : 0;

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
        else if (height >= 2 && depth < MAX_PLY && searchInfo->killer[height - 2][0] == *ptr)
            movePrice[height][i] = 99999000000llu;
        else if (depth < MAX_PLY && searchInfo->killer[height][1] == *ptr)
            movePrice[height][i] = 99998000000llu;
        else if (height >= 2 && depth < MAX_PLY && searchInfo->killer[height - 2][1] == *ptr)
            movePrice[height][i] = 99997000000llu;
        else {
            U16 mto = MoveTo(*ptr);
            U8 mpc = board->squares[MoveFrom(*ptr)];
            movePrice[height][i] = history[board->color][MoveFrom(*ptr)][mto]
                                 + contHist[pp1][pt1][mpc][mto]
                                 + contHist2[pp2][pt2][mpc][mto];
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

    initSearchParams();
    refreshDerivedTables();

    clearHistory();
}

// Defaults for the tunable search parameters (overridable via UCI setoption).
void initSearchParams() {
    FutilityStep = 103;
    ReverseFutilityStep = 47;
    RazorMargin = 297;
    LmpPruningAllow = 1;   // LMP enabled (tuned in the package)
    LmpMaxDepth = 8;
    LmpBase = 6;
    NmpBase = 4;
    NmpDiv = 4;
    AspirationDelta = 35;
    LmrBaseX100 = 71;    // 0.71
    LmrDivX100 = 187;    // 1.87
    LmrHistoryDiv = 20567;
    // adaptive time-management coefficients
    TmStabPct = 6;
    TmChangeX100 = 154;
    TmPanicX100 = 133;
    TmPanicDrop = 24;
    TmMaxMult = 4;
}

// Rebuild depth/move-count tables that depend on the tunables. Call after any
// setoption that changes LmrBaseX100/LmrDivX100/LmpBase.
void refreshDerivedTables() {
    double lmrBase = LmrBaseX100 / 100.0;
    double lmrDiv = LmrDivX100 / 100.0;
    for (int i = 1; i < MAX_PLY; ++i) {
        for (int j = 1; j < 64; ++j) {
            lmr[i][j] = lmrBase + log(i) * log(j) / lmrDiv;
        }
    }

    // Late Move Pruning thresholds: prune more aggressively when not improving.
    for (int d = 0; d < MAX_PLY; ++d) {
        lmpTable[1][d] = LmpBase + d * d;         // improving
        lmpTable[0][d] = (LmpBase + d * d) / 2;   // not improving
    }
}

void resetSearchInfo(SearchInfo *info, TimeManager tm) {
    memset(info, 0, sizeof(SearchInfo));
    info->tm = tm;
    setAbort(0);
}

void setAbort(int val) {
    pthread_mutex_lock(&mutex);
    ABORT = val;
    pthread_mutex_unlock(&mutex);
}

void clearHistory() {
    memset(history, 0, sizeof(history));
    memset(contHist, 0, sizeof(contHist));
    memset(contHist2, 0, sizeof(contHist2));
}

