#include <assert.h>

#include "mcts.h"
#include "search.h"

struct MCTSNode {
    U16 move;
    MCTSNode **children;
    int childrenCount;
    double w;
    double n;
};

MCTSNode *createMCTSNode(U16 move) {
    MCTSNode *node = (MCTSNode *) malloc(sizeof(MCTSNode));

    node->move = move;
    node->n = 0;
    node->w = 0;
    node->children = NULL;
    node->childrenCount = 0;

    return node;
}

U16 movesCash1[512];
U16 movesCash2[512];
MCTSNode *nodesStack[65536];
Undo undoStack[65536];
U16 undoMove[65536];

const double C = 1;

MCTSNode *choseMaxChildren(MCTSNode *parent, int print) {
    assert(parent->children && parent->childrenCount);

    MCTSNode *res = parent->children[0];

    double maxWeight = 0;
    for (int i = 0; i < parent->childrenCount; ++i) {
        double w = parent->children[i]->w;
        double n = parent->children[i]->n;
        double wght = w / n + C * sqrt(log(parent->n - n) / n);

        if (print) {
            char *mv = getMove(parent->children[i]->move);
            printf("W/N: %f/%f, Weight: %f, Sibles: %f; MV: %s\n", w, n, wght, parent->n - n, mv);
            printf("%f >= %f\n", wght, maxWeight);
            free(mv);
        }

        if (wght >= maxWeight) {
            res = parent->children[i];
            maxWeight = wght;
        }

    }

    if (print) {
        char *mv = getMove(res->move);
        printf("MV: %s\n", mv);
        free(mv);
    }

    return res;
}

MCTSNode *getBest(MCTSNode *node) {
    double max = 0;
    MCTSNode *res = NULL;

    for (int i = 0; i < node->childrenCount; ++i) {
        if (node->children[i]->n >= max) {
            max = node->children[i]->n;
            res = node->children[i];
        }
    }

    return res;
}

int MCTSSearch(Board *board, TimeManager tm) {
    MCTSNode *root = createMCTSNode(0);

    MCTSNode *current;
    int stackIndex;
    int undoStackIndex;
    setAbort(0);
    Timer timer;
    startTimer(&timer);


    clock_t pvInterval = clock();
    clock_t testAbortInterval = clock();

    while (1) {
        // Step 0: Preparation
        stackIndex = 1;
        undoStackIndex = 0;
        current = root;
        nodesStack[0] = current;

        // Step 1: Choosing
        while (current->childrenCount) {
            nodesStack[stackIndex] = current;
            current = choseMaxChildren(current, 0);

            if (current->move) {
                makeMove(board, current->move, &undoStack[undoStackIndex]);
                undoMove[undoStackIndex] = current->move;
                undoStackIndex++;
            }

            ++stackIndex;
        }
        stackIndex--;

        // Step 2: Expansion
        runSimulationsForNode(board, current);

        // Step 3: Backpropogation
        int negativeScore = 0;

        double w = current->w;
        double n = current->n;
        double negativeW = n - w;

        while (stackIndex >= 0) {
            nodesStack[stackIndex]->n += n;
            nodesStack[stackIndex]->w += negativeScore ? negativeW : w;
            stackIndex--;
            negativeScore = !negativeScore;
        }

        while(undoStackIndex > 0) {
            undoStackIndex--;
            unmakeMove(board, undoMove[undoStackIndex], &undoStack[undoStackIndex]);
        }

        // Step 4: printing

        if ((clock() - pvInterval) / CLOCKS_PER_SEC > 0.5) {
            MCTSNode *pvNode = root;

            printf("info pv ");
            while (pvNode) {
                pvNode = getBest(pvNode);

                if (pvNode) {
                    char *mv = getMove(pvNode->move);
                    printf("%s ", mv);
                    free(mv);
                }
            }
            printf("\n");
            fflush(stdout);
            pvInterval = clock();
        }

        if ((clock() - testAbortInterval) > 0.001 &&  testAbort(getTime(&timer), 0, &tm) || SEARCH_COMPLETE) {
            testAbortInterval = clock();
            setAbort(1);

            MCTSNode *best = getBest(root);
            char *mv = getMove(best->move);
            __sync_synchronize();
            printf("bestmove %s\n", mv);
            fflush(stdout);
            SEARCH_COMPLETE = 1;
            free(mv);

            return 0;
        }
    }
}

void runSimulationsForNode(Board *board, MCTSNode *node) {
    int movesCount = generatePossibleMoves(board, movesCash1);

    node->children = malloc(sizeof(MCTSNode *) * movesCount);
    node->childrenCount = movesCount;

    for (int i = 0; i < movesCount; ++i) {
        node->children[i] = createMCTSNode(movesCash1[i]);
        double incW = simulate(board, movesCash2);

        double incN = 1;
        node->children[i]->n += incN;
        node->children[i]->w += incW;
        node->children[i]->move = movesCash1[i];
        node->n += incN;
        node->w += incW;
    }
}

double simulate(Board *board, U16 *movesCash) {
    if (isDraw(board)) {
        return 0.5;
    }

    int movesCount = generatePossibleMoves(board, movesCash);

    if (movesCount == 0) {
        return inCheck(board, board->color) ? 0 : 0.5;
    }

    int randMove = rand() % movesCount;

    Undo undo;
    U16 move = movesCash[randMove];
    // printMove(move);
    makeMove(board, move, &undo);
    double res = 1 - simulate(board, movesCash);
    unmakeMove(board, move, &undo);
    return res;
}

int generatePossibleMoves(Board *board, U16 *moves) {
    movegen(board, moves);

    U16 *curMove = moves;
    U16 *possibleMove = curMove;
    Undo undo;

    int movesCount = 0;
    while (*curMove) {
        makeMove(board, *curMove, &undo);

        if (!inCheck(board, !board->color)) {
            possibleMove[movesCount] = *curMove;
            ++movesCount;
        }

        unmakeMove(board, *curMove, &undo);
        curMove++;
    }

    possibleMove[movesCount] = 0;
    return movesCount;
}