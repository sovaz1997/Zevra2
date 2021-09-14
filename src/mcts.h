#include "board.h"

typedef struct MCTSNode MCTSNode;

MCTSNode* createMCTSNode(U16 move);
void runSimulationsForNode(Board* board, MCTSNode* node);
void addChild(MCTSNode* parent, MCTSNode* child);
int MCTSSearch();
int generatePossibleMoves(Board* board, U16* moves);
double simulate(Board* board, U16* movesCash);