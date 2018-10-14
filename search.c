#include "search.h"

void iterativeDeeping(Board* board, TimeManager tm) {
    SearchInfo searchInfo;
    char bestMove[6];
    
    resetSearchInfo(&searchInfo, tm);

    for(int i = 1; i <= tm.depth; ++i) {
        startTimer(&searchInfo.timer);
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
    if(isDraw(board) && height) {
        return 0;
    }

    if(!depth) {
        return quiesceSearch(board, searchInfo, alpha, beta, height);
    }

    int root = (height ? 0 : 1);

    if(depth >= 1) {
        if(searchInfo->tm.searchType == FixedTime) {
            if(getTime(&searchInfo->timer) >= searchInfo->tm.time) {
                searchInfo->abort = 1;
                return 0;
            }
        }
    }

    ++searchInfo->nodesCount;

    movegen(board, moves[height]);
    moveOrdering(board, moves[height], searchInfo, height);

    U16* curMove = moves[height];
    Undo undo;

    int movesCount = 0;
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
        
        ++movesCount;

        int eval = -search(board, searchInfo, -beta, -alpha, depth - 1, height + 1);        
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

            searchInfo->killer[board->color][height] = *curMove;
        }
        if(alpha >= beta) {
            break;
        }
        ++curMove;
    }

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
    int i;

    for(i = 0; *ptr; ++i) {
        movePrice[i] = 0;
        U16 toPiece = pieceType(board->squares[MoveTo(*ptr)]);
        U16 fromPiece = pieceType(board->squares[MoveFrom(*ptr)]);
        
        if(toPiece) {
            movePrice[i] = mvvLvaScores[fromPiece][toPiece];
        }
        if(searchInfo->bestMove == *ptr && !height) {
            movePrice[i] = 1000;
        }
        if(searchInfo->killer[board->color][height] == *ptr) {
            movePrice[i] = 100;
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
            } else if(victimScore == BISHOP) {
                victimScore = 30;
            } else if(victimScore == KNIGHT) {
                victimScore = 20;
            } else if(victimScore == PAWN) {
                victimScore = 10;
            }

            mvvLvaScores[attacker][victim] = victimScore - attacker;
        }
    }
}

void resetSearchInfo(SearchInfo* info, TimeManager tm) {
    memset(info, 0, sizeof(SearchInfo));
    info->tm = tm;
}