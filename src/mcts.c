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

int MCTSSearch(Board* board) {
    MCTSNode* root = createMCTSNode(0);
    runSimulationsForNode(board, root);

    for (int i = 0; i < root->childrenCount; ++i) {
        printf("%f\n", root->children[i]->w);
    }
}

void runSimulationsForNode(Board* board, MCTSNode* node) {
    int movesCount = generatePossibleMoves(board, movesCash1);

    node->children = malloc(sizeof(MCTSNode*) * movesCount);
    node->childrenCount =movesCount;

    for (int i = 0; i < movesCount; ++i) {
        node->children[i] = createMCTSNode(movesCash1[i]);
        node->children[i]->n++;
        node->children[i]->w += simulate(board, movesCash2);
        node->children[i]->move = movesCash1[i];
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
    printMove(move);
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