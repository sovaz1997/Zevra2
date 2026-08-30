#ifndef SEARCH_H
#define SEARCH_H

#include <math.h>
#include <time.h>
#include <pthread.h>
#include "types.h"
#include "board.h"
#include "movegen.h"
#include "eval.h"
#include "uci.h"
#include "timemanager.h"
#include "transposition.h"

extern volatile int ABORT;
extern volatile int SEARCH_COMPLETE;

struct SearchArgs {
    Board* board;
    TimeManager tm;
};

extern long long history[2][64][64];
// Continuation history: [prevPiece][prevTo][curPiece][curTo], raw piece codes 0..13.
// contHist = 1-ply (counter-move), contHist2 = 2-ply. Updated with bonus (cutoff
// move) and malus (searched-but-failed quiets), gravity-bounded to +/-MAX_HISTORY.
#define MAX_HISTORY 16384
extern long long contHist[14][64][14][64];
extern long long contHist2[14][64][14][64];
extern U8 contPiece[MAX_PLY + 2];   // moving piece code of the move descending from each height
extern U8 contTo[MAX_PLY + 2];      // its destination square

struct SearchInfo {
    U64 nodesCount;
    U16 bestMove;
    Timer timer;
    TimeManager tm;
    U16 killer[MAX_PLY + 1][2];
    int nullMoveSearch;
    int searchTime;
    int selDepth;
    int eval;
};

//Eval type
enum {
    empty = 0,
    lowerbound = 1,
    upperbound = 2,
    exact = 3
};

// Tunable search parameters. Declared without an initialiser (tentative defs
// merged by -fcommon, like the tables below) and given defaults at runtime in
// initSearchParams(); overridable via UCI setoption for SPSA tuning.
extern int FutilityStep;         // futility margin per depth
extern int ReverseFutilityStep;  // reverse-futility (static null) margin per depth
extern int RazorMargin;          // razoring margin per depth
extern int LmpMaxDepth;          // apply LMP only at depth <= this
extern int LmpBase;              // move-count base: lmpTable[imp][d] = LmpBase + d*d (halved if !improving)
extern int NmpBase;              // null-move reduction R = NmpBase + depth / NmpDiv
extern int NmpDiv;
extern int AspirationDelta;      // initial aspiration window half-width
extern int LmrBaseX100;          // LMR: lmr[i][j] = LmrBaseX100/100.0 + log(i)*log(j)/(LmrDivX100/100.0)
extern int LmrDivX100;
extern int LmrHistoryDiv;        // selective LMR: reduction -= clamp(quietHistory/LmrHistoryDiv, -2, 2)
extern int LmpPruningAllow;      // 0 = LMP off (champion default); 1 = on (SPSA candidate)

extern U16 moves[MAX_PLY][256];
extern int improving[256];
extern U8 temperatureOffsets[256];
extern long long movePrice[MAX_PLY][256];
extern int mvvLvaScores[7][7];
extern int lmr[MAX_PLY][64];
// Late Move Pruning move-count thresholds: lmpTable[improving][depth].
extern int lmpTable[2][MAX_PLY];

//Heuristics control
static const int LmrPruningAllow = 1;

// Search statistics (compile with -DSEARCH_STATS to enable; zero cost otherwise).
// Read via `bench` to inspect how each heuristic behaves inside the tree.
typedef struct SearchStats {
    U64 nodes;            // main-search nodes (denominator for fire rates)
    U64 failHigh;         // nodes that produced a beta cutoff
    U64 failHighFirst;    // ...of which the first legal move caused the cutoff
    U64 nmpTries;         // null-move attempts
    U64 nmpCut;           // ...that returned a cutoff
    U64 rfpCut;           // reverse-futility prunes (returns)
    U64 razorCut;         // razoring drops to qsearch
    U64 futilityCut;      // futility prunes of a move
    U64 lmrTries;         // reduced (LMR) searches performed
    U64 lmrReSearch;      // ...that failed high and needed a full re-search
    U64 lmpCut;           // late-move (move-count) prunes of a quiet move
    U64 aspTries;         // aspiration-window searches
    U64 aspFail;          // ...that fell outside the window (re-search)
    U64 lmrVerChecked;    // VERIFY_LMR: reduced+skipped moves re-checked at full depth
    U64 lmrVerMissed;     // ...that actually beat alpha at full depth (blind over-reduction)
} SearchStats;

extern SearchStats g_stats;

#ifdef SEARCH_STATS
    #define STAT(x) (g_stats.x++)
#else
    #define STAT(x) ((void) 0)
#endif

#define ENABLE_REVERSE_FUTILITY_PRUNING 1
#define ENABLE_RAZORING 1
#define ENABLE_NULL_MOVE_PRUNING 1
#define ENABLE_FUTILITY_PRUNING 1
// LMP is compiled in but gated at runtime by LmpPruningAllow (default 0 = off), so
// the champion is unchanged and pays only a predictable short-circuited branch.
// SPSA turns it on (LmpPruningAllow=1) and tunes LmpMaxDepth/LmpBase; flip the
// LmpPruningAllow default to 1 only after a tuned LMP passes an SPRT.
#ifndef ENABLE_LATE_MOVE_PRUNING
#define ENABLE_LATE_MOVE_PRUNING 1
#endif

void* go(void* thread_data);
SearchInfo iterativeDeeping(Board* board, TimeManager tm);
int search(Board* board, SearchInfo* searchInfo, int alpha, int beta, int depth, int height);
int aspirationWindow(Board* board, SearchInfo* searchInfo, int depth, int score);
int quiesceSearch(Board* board, SearchInfo* searchInfo, int alpha, int beta, int height);
U64 perftTest(Board* board, int depth, int height);
void perft(Board* board, int depth);
void* perftThreads(void* perftArgs);
void moveOrdering(Board* board, U16* mvs, SearchInfo* searchInfo, int height, int depth);
void initSearch();
void initSearchParams();        // set tunable-parameter defaults
void refreshDerivedTables();    // rebuild lmr[]/lmpTable[] from the tunables
void resetSearchInfo(SearchInfo* info, TimeManager tm);
void setAbort(int val);
void clearHistory();
void movePick(int moveNumber, int height);
void resetSearchStats();
void printSearchStats();

#endif