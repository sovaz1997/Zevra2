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

            if(!strcmp(go_param, "perft")) {
                char* depth_str = strtok_r(NULL, " ", &context);
                perft(board, atoi(depth_str));
            } else if(!strcmp(go_param, "depth")) {
                char* depth_str = strtok_r(NULL, " ", &context);
                iterativeDeeping(board, createFixDepthTm(atoi(depth_str)));
            } else if(!strcmp(go_param, "movetime")) {
                char* time_str = strtok_r(NULL, " ", &context);
                iterativeDeeping(board, createFixTimeTm(atoll(time_str)));
            } else if(!strcmp(go_param, "infinite")) {
                char* time_str = strtok_r(NULL, " ", &context);
                iterativeDeeping(board, createFixDepthTm(MAX_PLY));
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
        printf("%s ", mv);
        makeMove(board, cur->move, &undo);
        cur = &tt[board->key & ttIndex];
    }
    *board = b;
    board->gameInfo->moveCount = moveCount;
}