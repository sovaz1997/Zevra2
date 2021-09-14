#include <assert.h>

#include "mcts.h"
#include "search.h"

struct MCTSNode {
    U16 move;
    MCTSNode** children;
    int childrenCount;
    double w;
    double n;
};

MCTSNode* createMCTSNode(U16 move) {
    MCTSNode* node = (MCTSNode*)malloc(sizeof(MCTSNode));

    node->move = move;
    node->n = 0;
    node->w = 0;
    node->children = NULL;
    node->childrenCount = 0;

    return node;
}

U16 movesCash1[512];
U16 movesCash2[512];
MCTSNode* nodesStack[65536];

const double C = 1;

MCTSNode* choseMaxChildren(MCTSNode* parent, int print) {
    assert(parent->children && parent->childrenCount);

    MCTSNode * res = parent->children[0];

    double maxWeight = 0;
    for (int i = 0; i < parent->childrenCount; ++i) {
        double w = parent->children[i]->w;
        double n = parent->children[i]->n;
        double wght = w / n + C * sqrt(log(parent->n - n) / n);

        if (print) {
            char* mv = getMove(parent->children[i]->move);
            printf("W/N: %f/%f, Weight: %f, Sibles: %f; MV: %s\n", w, n, wght, parent->n - n, mv);
            printf("%f >= %f\n", wght, maxWeight);
            free(mv);
        }

        if (wght >= maxWeight) {
            res = parent->children[i];
            maxWeight = wght;
            // printf("%f/%f: %f %f\n", w, n, weight, sqrt(log(parent->n) / n));
        }

    }

    if (print) {
        char* mv = getMove(res->move);
        printf("MV: %s\n", mv);
        free(mv);
    }

    return res;
}

int MCTSSearch(Board* board) {
    MCTSNode* root = createMCTSNode(0);

    MCTSNode* current;
    int stackIndex;

    while(1) {
        // Step 0: Preparation
        stackIndex = 1;
        current = root;
        nodesStack[0] = current;

        // Step 1: Choosing
        while (current->childrenCount) {
            nodesStack[stackIndex] = current;
            current = current == root ? choseMaxChildren(current, 0) : choseMaxChildren(current, 0);

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

        while(stackIndex >= 0) {
            nodesStack[stackIndex]->n += n;
            nodesStack[stackIndex]->w += negativeScore ? negativeW : w;
            stackIndex--;
            negativeScore = !negativeScore;
        }

        // Step 4: printing

        for (int i = 0; i < root->childrenCount; ++i) {
            char* mv = getMove(root->children[i]->move);
            printf("%s: %f/%f\n", mv, root->children[i]->w, root->children[i]->n);
            free(mv);
        }
    }
/*
    runSimulationsForNode(board, root);

    for (int i = 0; i < root->childrenCount; ++i) {
        printf("%f\n", root->children[i]->w);
    }*/
}

void runSimulationsForNode(Board* board, MCTSNode* node) {
    int movesCount = generatePossibleMoves(board, movesCash1);

    node->children = malloc(sizeof(MCTSNode*) * movesCount);
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

double simulate(Board* board, U16* movesCash) {
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

int generatePossibleMoves(Board* board, U16* moves) {
    movegen(board, moves);

    U16* curMove = moves;
    U16* possibleMove = curMove;
    Undo undo;

    int movesCount = 0;
    while(*curMove) {
        makeMove(board, *curMove, &undo);

        if(!inCheck(board, !board->color)) {
            possibleMove[movesCount] = *curMove;
            ++movesCount;
        }

        unmakeMove(board, *curMove, &undo);
        curMove++;
    }

    possibleMove[movesCount] = 0;
    return movesCount;
}