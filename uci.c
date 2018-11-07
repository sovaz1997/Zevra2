#include "uci.h"

char startpos[] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
Option option;

int main() {
    initOption();
    initEngine();

    Board* board = (Board*) malloc(sizeof(Board)); 

    printEngineInfo();
    setFen(board, startpos);

    char buff[65536];
    char* context;

    GameInfo gameInfo;
    gameInfo.moveCount = 0;
    board->gameInfo = &gameInfo;
    SEARCH_COMPLETE = 1;

    while(1) {
        input(buff);

        char* str = strdup(buff);
        
        char* fen = strstr(str, "fen") + 4;
        char* startposStr = strstr(str, "startpos");
        char* moves = strstr(str, "moves");
        char* name = strstr(str, "name") + 5;
        char* value = strstr(str, "value") + 6;
        
        if(moves) {
            moves += strlen("moves ");
        }

        char* cmd = strtok_r(str, " ", &context);

        if(!strcmp(cmd, "go") && SEARCH_COMPLETE) {
            char* go_param = strtok_r(NULL, " ", &context);
            if(!strcmp(go_param, "perft")) {
                char* depth_str = strtok_r(NULL, " ", &context);
                perft(board, atoi(depth_str));
            } else {
                TimeManager tm;
                if(!strcmp(go_param, "depth")) {
                    char* depth_str = strtok_r(NULL, " ", &context);
                    tm = createFixDepthTm(atoi(depth_str));
                } else if(!strcmp(go_param, "movetime")) {
                    char* time_str = strtok_r(NULL, " ", &context);
                    tm = createFixTimeTm(atoll(time_str));
                } else if(!strcmp(go_param, "infinite")) {
                    char* time_str = strtok_r(NULL, " ", &context);
                    tm = createFixDepthTm(MAX_PLY);
                } else {
                    int wtime = 0, btime = 0, winc = 0, binc = 0, movestogo = 0;

                    while(1) {
                        if(!go_param) {
                            break;
                        }

                        if(!strcmp(go_param, "wtime")) {
                            char* tm = strtok_r(NULL, " ", &context);
                            wtime = atoi(tm);
                        } else if(!strcmp(go_param, "btime")) {
                            char* tm = strtok_r(NULL, " ", &context);
                            btime = atoi(tm);
                        } else if(!strcmp(go_param, "winc")) {
                            char* inc = strtok_r(NULL, " ", &context);
                            winc = atoi(inc);
                        } else if(!strcmp(go_param, "binc")) {
                            char* inc = strtok_r(NULL, " ", &context);
                            binc = atoi(inc);
                        } else if(!strcmp(go_param, "movestogo")) {
                            char* mtg = strtok_r(NULL, " ", &context);
                            movestogo = atoi(mtg);
                        }

                        go_param = strtok_r(NULL, " ", &context);
                    }

                    tm = createTournamentTm(board, wtime, btime, winc, binc, movestogo);
                }

                SearchArgs args;
                args.board = board;
                args.tm = tm;
                
                pthread_t searchThread;
                SEARCH_COMPLETE = 0;
                pthread_create(&searchThread, NULL, &go, &args);
                //iterativeDeeping(args.board, args.tm);
                
            }
        } else if(!strcmp(cmd, "position") && SEARCH_COMPLETE) {
            gameInfo.moveCount = 0;
            cmd = strtok_r(NULL, " ", &context);
            int cmd_success_input = 0;
            if(startposStr) {
                setFen(board, startpos);
                cmd_success_input = 1;
            } else if(fen) {
                setFen(board, fen);
                cmd_success_input = 1;
            }
            
            if(cmd_success_input) {
                setMovesRange(board, moves);
            }

        } else if(!strcmp(cmd, "d") && SEARCH_COMPLETE) {
            printBoard(board);
        } else if(!strcmp(cmd, "quit")) {
            free(str);
            free(tt);
            break;
        } else if(!strcmp(cmd, "uci") && SEARCH_COMPLETE) {
            printEngineInfo();
            printf("option name Hash type spin default %d min %d max %d\n", option.defaultHashSize, option.minHashSize, option.maxHashSize);
            printf("option name Clear Hash type button\n");
            printf("uciok\n");
        } else if(!strcmp(cmd, "eval") && SEARCH_COMPLETE) {
            printf("Eval: %d\n", fullEval(board));
        } else if(!strcmp(cmd, "isready")) {
            readyok();
        } else if(!strcmp(cmd, "stop") && !SEARCH_COMPLETE) {
            SEARCH_COMPLETE = 1;
            setAbort(1);
        } else if(!strcmp(cmd, "ucinewgame") && SEARCH_COMPLETE) {
            clearTT();
        } else if(!strcmp(cmd, "setoption") && SEARCH_COMPLETE) {
            if(name && value) {
                if(!strncmp(name, "Hash", 4)) {
                    int hashSize = atoi(value);
                    if(hashSize >= option.minHashSize && hashSize <= option.maxHashSize) {
                        reallocTT(hashSize);
                    }
                    printf("info string hash size changed to %d mb\n", hashSize);
                } else if(!strncmp(name, "Clear Hash", 10)) {
                    clearTT();
                    printf("info string hash cleared\n");
                }
            }
        }

        fflush(stdout);

        free(str);
    }

    free(board);
}

void printEngineInfo() {
    printf("id name Zevra v2.0 r172\nid author Oleg Smirnov\n");    
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
        if(score < 0) {
            printf("score mate %d", (-MATE_SCORE - score) / 2);
        } else {
            printf("score mate %d", (MATE_SCORE - score + 1) / 2);
        }
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
    fgets(str, 65536, stdin);
    char* ptr = strchr(str, '\n');
    if (ptr) {
        *ptr = '\0';
    } 
    ptr = strchr(str, '\r');
    if (ptr) {
        *ptr = '\0';
    }
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
    
    for(int i = 0; (cur->evalType == lowerbound || cur->evalType == exact) && !isDraw(board) && i < depth + 20; ++i) {
        char mv[6];
        moveToString(cur->move, mv);

        if(findMove(mv, board)) {
            makeMove(board, cur->move, &undo);
            if(inCheck(board, !board->color)) {
                break;
            }
            printf("%s ", mv);
        } else {
            break;
        }
        cur = &tt[board->key & ttIndex];
    }
    *board = b;
    board->gameInfo->moveCount = moveCount;
}

int findMove(char* move, Board* board) {
    U16 moves[256];
    movegen(board, moves);

    U16* curMove = moves;
    while(*curMove) {
        char mv[6];
        moveToString(*curMove, mv);
        if(!strcmp(move, mv)) {
            return 1;
        }
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