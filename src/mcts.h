#include "board.h"

typedef struct MCTSNode MCTSNode;

MCTSNode* createMCTSNode(U16 move);
MCTSNode* choseMaxChildren(MCTSNode* parent, int print);
void runSimulationsForNode(Board* board, MCTSNode* node);
int MCTSSearch(Board* board);
int generatePossibleMoves(Board* board, U16* moves);
double simulate(Board* board, U16* movesCash);