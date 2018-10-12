#include "uci.h"

char startpos[] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

void uciInterface(Board* board) {
    printf("id name Zevra v2.0 dev\nid author Oleg Smirnov\n");
    setFen(board, startpos);

    char buff[65536];
    char* context;

    while(1) {
        scanf("%[^\n]%*c", buff);
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
            }
        } else if(!strcmp(cmd, "position")) {
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
                char* moveRange = strstr(cmd, "moves") + strlen("moves ");
                setMovesRange(board, moves);
            }

        } else if(!strcmp(cmd, "d")) {
            printBoard(board);
        }
        free(str);
    }
}