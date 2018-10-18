#include "uci.h"

char startpos[] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

void uciInterface(Board* board) {
    printEngineInfo();
    setFen(board, startpos);

    char buff[65536];
    char* context;

    GameInfo gameInfo;
    gameInfo.moveCount = 0;
    board->gameInfo = &gameInfo;

    while(1) {
        input(buff);

        char* str = strdup(buff);
        
        char* fen = strstr(str, "fen");
        if(fen) {
            fen += strlen("fen ");
        }
        char* startposStr = strstr(str, "startpos");

        char* moves = strstr(str, "moves");
        if(moves) {
            moves += strlen("moves ");
        }

        char* cmd = strtok_r(str, " ", &context);

        if(!strcmp(cmd, "go")) {
            char* go_param = strtok_r(NULL, " ", &context);
            TimeManager tm;
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
                            binc = atoi(mtg);
                        }

                        go_param = strtok_r(NULL, " ", &context);
                    }

                    tm = createTournamentTm(board, wtime, btime, winc, binc, movestogo);
                }

                SearchArgs args;
                args.board = board;
                args.tm = tm;
                
                runSearch(&args);
                //pthread_create(&searchThread, NULL, &go, &args);
            }
        } else if(!strcmp(cmd, "position")) {
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

        } else if(!strcmp(cmd, "d")) {
            printBoard(board);
        } else if(!strcmp(cmd, "quit")) {
            free(str);
            free(tt);
            break;
        } else if(!strcmp(cmd, "uci")) {
            printEngineInfo();
            printf("uciok\n");
        } else if(!strcmp(cmd, "eval")) {
            printf("Eval: %d\n", fullEval(board));
        } else if(!strcmp(cmd, "isready")) {
            printf("readyok\n");
        }

        fflush(stdout);

        free(str);
    }
}

void runSearch(SearchArgs* args) {
    pthread_t searchThread;
    pthread_create(&searchThread, NULL, &go, args);
    char buff[9];
    while(1) {
        input(buff);

        if(!strcmp(buff, "stop")) {
            pthread_mutex_lock(&mutex);
            ABORT = 1;
            pthread_mutex_unlock(&mutex);
            return;
        }
    }
}

void printEngineInfo() {
    printf("id name Zevra v2.0 dev\nid author Oleg Smirnov\n");
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

void printPV(Board* board, int depth) {
    Transposition* cur = &tt[board->key & ttIndex];

    int moveCount = board->gameInfo->moveCount;
    Board b = *board;
    Undo undo;

    
    for(int i = 0; (cur->evalType == lowerbound || cur->evalType == exact) && !isDraw(board) && i < depth + 20; ++i) {
        char mv[6];
        moveToString(cur->move, mv);

        
        if(findMove(mv, board)) {
            printf("%s ", mv);
        } else {
            break;
        }
        makeMove(board, cur->move, &undo);
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