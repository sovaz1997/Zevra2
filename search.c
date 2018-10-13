#include "search.h"

void iterativeDeeping(Board* board, int depth) {
    SearchInfo searchInfo;
    char bestMove[6];
    for(int i = 1; i <= depth; ++i) {
        memset(&searchInfo, 0, sizeof(SearchInfo));
        int eval = search(board, &searchInfo, -MATE_SCORE, MATE_SCORE, i, 0);
        moveToString(searchInfo.bestMove, bestMove);
        printf("info depth %d nodes %d ", i, searchInfo.nodesCount);
        printScore(eval);
        printf(" pv %s\n", bestMove);
    }

    printf("bestmove %s\n", bestMove);
}

int search(Board* board, SearchInfo* searchInfo, int alpha, int beta, int depth, int height) {
    if(!depth) {
        ++searchInfo->nodesCount;
        return fullEval(board);
    }

    movegen(board, moves[height]);

    U16* curMove = moves[height];
    Undo undo;

    int movesCount = 0;
    while(*curMove) {
        makeMove(board, *curMove, &undo);

        if(!inCheck(board, !board->color)) {
            ++movesCount;

            int eval = -search(board, searchInfo, -beta, -alpha, depth - 1, height + 1);

            if(eval > alpha) {
                alpha = eval;
                if(!height) {
                    searchInfo->bestMove = *curMove;
                }
            }
            if(alpha >= beta) {
                unmakeMove(board, *curMove, &undo);
                break;
            }
        }

        unmakeMove(board, *curMove, &undo);

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