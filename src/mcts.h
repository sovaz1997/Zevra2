#include "board.h"

typedef struct MCTSNode MCTSNode;
MCTSNode* getBest(MCTSNode* node);
MCTSNode* createMCTSNode(U16 move);
MCTSNode* choseMaxChildren(MCTSNode* parent, int print);
void runSimulationsForNode(Board* board, MCTSNode* node);
int MCTSSearch(Board* board, TimeManager tm);
int generatePossibleMoves(Board* board, U16* moves);
double simulate(Board* board, U16* movesCash);