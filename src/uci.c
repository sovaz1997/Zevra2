#include "uci.h"

char startpos[] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
Option option;

const int TUNING_ENABLED = 0;


int main(int argc, char** argv) {

    initOption();
    initEngine();

    Board* board = (Board*) malloc(sizeof(Board));
    nnue = (NNUE*) malloc(sizeof(NNUE));

//    if (argc > 2) {
//        if (strEquals(argv[1], "--weights-file")) {
//            loadWeights(argv[2]);
//        }
//    }

    NNUE* nnue = (NNUE*) malloc(sizeof(NNUE));
    loadNNUEWeights();

    printEngineInfo();
    setFen(board, startpos);

    // initNNUEPosition(nnue, board);

    char buff[65536];

    GameInfo gameInfo;
    gameInfo.moveCount = 0;
    board->gameInfo = &gameInfo;
    SEARCH_COMPLETE = 1;

    // tuning
//    if (TUNING_ENABLED) {
//        makeTuning(board);
//    }

    TimeManager tm = initTM();

    while(1) {
        int makeSearch = 0;
        input(buff);

        char* str = strdup(buff);
        
        char* fen = strstr(str, "fen") + 4;
        char* startposStr = strstr(str, "startpos");
        char* mvs = strstr(str, "moves");
        char* name = strstr(str, "name") + 5;
        char* value = strstr(str, "value") + 6;
        
        if(mvs)
            mvs += strlen("moves ");

        char* cmd = strtok(str, " ");

        if(!cmd)
            continue;
        __sync_synchronize();

        if(strStartsWith(cmd, "go") && SEARCH_COMPLETE) {
            char* go_param = strtok(NULL, " ");
            
            if(!go_param) {
                tm = createFixDepthTm(MAX_PLY - 1);
                makeSearch = 1;
            } else if(strEquals(go_param, "perft")) {
                char* depth_str = strtok(NULL, " ");
                if(!depth_str) {
                    continue;
                }
                perft(board, atoi(depth_str));
            } else {
                if(strEquals(go_param, "depth")) {
                    char* depth_str = strtok(NULL, " ");
                    tm = createFixDepthTm(atoi(depth_str));
                } else if(strEquals(go_param, "movetime")) {
                    char* time_str = strtok(NULL, " ");
                    tm = createFixTimeTm(atoll(time_str));
                } else if(strEquals(go_param, "infinite")) {
                    tm = createFixDepthTm(MAX_PLY - 1);
                } else if(strEquals(go_param, "nodes")) {
                    char* nodes_str = strtok(NULL, " ");
                    tm = createFixedNodesTm(atoi(nodes_str));
                } else {
                    int wtime = 0, btime = 0, winc = 0, binc = 0, movestogo = 0;
                    
                    while(go_param) {
                        if(strEquals(go_param, "wtime")) {
                            char* tm_str = strtok(NULL, " ");
                            wtime = atoi(tm_str);
                        } else if(strEquals(go_param, "btime")) {
                            char* tm_str = strtok(NULL, " ");
                            btime = atoi(tm_str);
                        } else if(strEquals(go_param, "winc")) {
                            char* inc = strtok(NULL, " ");
                            winc = atoi(inc);
                        } else if(strEquals(go_param, "binc")) {
                            char* inc = strtok(NULL, " ");
                            binc = atoi(inc);
                        } else if(strEquals(go_param, "movestogo")) {
                            char* mtg = strtok(NULL, " ");
                            movestogo = atoi(mtg);
                        }

                        go_param = strtok(NULL, " ");
                    }

                    tm = createTournamentTm(board, wtime, btime, winc, binc, movestogo);
                }
                makeSearch = 1;
            }
        } else if(strEquals(cmd, "position") && SEARCH_COMPLETE) {
            gameInfo.moveCount = 0;
            cmd = strtok(NULL, " ");
            int cmd_success_input = 0;
            if(startposStr) {
                setFen(board, startpos);
                cmd_success_input = 1;
            } else if(fen) {
                setFen(board, fen);
                cmd_success_input = 1;
            }
            
            if(cmd_success_input)
                setMovesRange(board, mvs);

        } else if(strEquals(cmd, "d") && SEARCH_COMPLETE) {
            printBoard(board);
        } else if(strEquals(cmd, "quit")) {
            free(str);
            free(tt);
            break;
        } else if(strEquals(cmd, "uci") && SEARCH_COMPLETE) {
            printEngineInfo();
            printf("option name Hash type spin default %d min %d max %d\n", option.defaultHashSize, option.minHashSize, option.maxHashSize);
            printf("option name Clear Hash type button\n");
            printf("uciok\n");
        } else if(strEquals(cmd, "eval") && SEARCH_COMPLETE) {
            printf("Eval: %d\n", fullEval(board));
        } else if(strEquals(cmd, "isready")) {
            readyok();
        } else if(strEquals(cmd, "stop") && !SEARCH_COMPLETE) {
            SEARCH_COMPLETE = 1;
            setAbort(1);
        } else if(strEquals(cmd, "ucinewgame") && SEARCH_COMPLETE) {
            clearTT();
            clearHistory();
        } else if(strEquals(cmd, "setoption") && SEARCH_COMPLETE) {
            if(name && value) {
                if(strStartsWith(name, "Hash")) {
                    int hashSize = atoi(value);
                    if(hashSize >= option.minHashSize && hashSize <= option.maxHashSize) {
                        reallocTT(hashSize);
                    }
                    printf("info string hash size changed to %d mb\n", hashSize);
                } else if(strStartsWith(name, "Clear Hash")) {
                    clearTT();
                    printf("info string hash cleared\n");
                }
            }
        }

        if(makeSearch) {
            SearchArgs args;
            args.board = board;
            args.tm = tm;
            
            pthread_t searchThread;
            SEARCH_COMPLETE = 0;
            pthread_create(&searchThread, NULL, &go, &args);
            pthread_detach(searchThread);
        }

        fflush(stdout);

        free(str);
    }

    destroyEval();
    free(board);
}

void printEngineInfo() {
    printf("id name Zevra v2.5\nid author Oleg Smirnov\n");
}

void readyok() {
    pthread_mutex_lock(&mutex);
    printf("readyok\n");
    pthread_mutex_unlock(&mutex);
}

void printScore(int score) {
    if(abs(score) < MATE_SCORE - 100) {
        printf("score cp %d", score);
    } else {
        if(score < 0)
            printf("score mate %d", (-MATE_SCORE - score) / 2);
        else
            printf("score mate %d", (MATE_SCORE - score + 1) / 2);
    }
}

void printSearchInfo(SearchInfo* info, Board* board, int depth, int eval, int evalType) {
    U64 searchTime = getTime(&info->timer);
    int speed = (searchTime < 1 ? 0 : (info->nodesCount / (searchTime / 1000.)));
    int hashfull = (double)ttFilledSize  / (double)ttSize * 1000;
    
    printf("info depth %d seldepth %d nodes %llu time %llu nps %d hashfull %d ", depth, info->selDepth, info->nodesCount, searchTime, speed, hashfull);
    printScore(eval);
    printf(evalType == lowerbound ? " lowerbound pv " : evalType == upperbound ? " upperbound pv " : " pv ");
    printPV(board, depth, info->bestMove);
    printf("\n");
    fflush(stdout);
}

void input(char* str) {
    if(!fgets(str, 65536, stdin))
        return;
    char* ptr = strchr(str, '\n');
    if (ptr)
        *ptr = '\0';
    ptr = strchr(str, '\r');
    if (ptr)
        *ptr = '\0';
}

void printPV(Board* board, int depth, U16 bestMove) {
    int moveCount = board->gameInfo->moveCount;
    Board b = *board;
    Undo undo;

    char mv[6];
    moveToString(bestMove, mv);
    makeMove(board, bestMove, &undo);
    printf("%s ", mv);

    Transposition* cur = &tt[board->key & ttIndex];
    
    for(int i = 0; !isDraw(board) && i < depth + 20; ++i) {
        if (!cur->evalType) {
            break;
        }

        moveToString(cur->move, mv);

        if(findMove(mv, board)) {
            makeMove(board, cur->move, &undo);
            if(inCheck(board, !board->color))
                break;
            printf("%s ", mv);
        } else
            break;
        cur = &tt[board->key & ttIndex];
    }
    *board = b;
    board->gameInfo->moveCount = moveCount;
}

int findMove(char* move, Board* board) {
    U16 mvs[256];
    movegen(board, mvs);

    U16* curMove = mvs;
    while(*curMove) {
        char mv[6];
        moveToString(*curMove, mv);
        if(strEquals(move, mv))
            return 1;
        ++curMove;
    }

    return 0;
}

void initEngine() {
    initBitboards();
    zobristInit();
    magicArraysInit();
    initSearch();
    initTT(option.defaultHashSize);
    initEval();
}

void initOption() {
    option.defaultHashSize = 256;
    option.minHashSize = 1;
    option.maxHashSize = 65536;
}

int strEquals(char* str1, char* str2) {
    return !strcmp(str1, str2);
}

int strStartsWith(char* str, char* key) {
    return strstr(str, key) == str;
}