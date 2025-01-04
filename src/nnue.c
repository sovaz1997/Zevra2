#include <stdio.h>
#include "nnue.h"
#include "timemanager.h"
#include "search.h"

int isExists(Board* board, int color, int piece, int sq) {
    return !!(board->pieces[piece] & board->colours[color] & bitboardCell(sq));
}

int getInputIndexOf(int color, int piece, int sq) {
    return color * 64 * 6 + (piece - 1) * 64 + sq;
}

double ReLU(double x) {
    return x > 0 ? x : 0;
}

int counter = 0;

void setNNUEInput(NNUE* nnue, int index) {
    if (nnue->inputs[index] == 1) {
        return;
    }

    nnue->inputs[index] = 1;

    // Update eval
    for (int i = 0; i < INNER_LAYER_COUNT; ++i) {
        nnue->accumulators[i] += nnue->weights_1[i][index];
    }

    nnue->eval = 0;

    // init array
    for (int i = 0; i < INNER_LAYER_COUNT; ++i) {
        nnue->accumulators2[i] = 0;
    }
    for (int i = 0; i < INNER_LAYER_COUNT; ++i) {
        for(int j = 0; j < INNER_LAYER_COUNT; ++j) {
            nnue->accumulators2[i] += ReLU(nnue->accumulators[j]) * nnue->weights_2[i][j];
        }
    }

    for (int i = 0; i < INNER_LAYER_COUNT; ++i) {
        nnue->eval += ReLU(nnue->accumulators2[i]) * nnue->weights_3[i];
    }
    //printf("Eval: %f\n", nnue->eval);
}

void resetNNUEInput(NNUE* nnue, int index) {
    if (!nnue->inputs[index]) {
        return;
    }

    nnue->inputs[index] = 0;

    counter--;

    // Update eval
    for (int i = 0; i < INNER_LAYER_COUNT; ++i) {
        nnue->accumulators[i] -= nnue->weights_1[i][index];
    }

    nnue->eval = 0;

    // init array
    for (int i = 0; i < INNER_LAYER_COUNT; ++i) {
        nnue->accumulators2[i] = 0;
    }
    for (int i = 0; i < INNER_LAYER_COUNT; ++i) {
      for(int j = 0; j < INNER_LAYER_COUNT; ++j) {
          nnue->accumulators2[i] += ReLU(nnue->accumulators[j]) * nnue->weights_2[i][j];
      }
    }

    for (int i = 0; i < INNER_LAYER_COUNT; ++i) {
        nnue->eval += ReLU(nnue->accumulators2[i]) * nnue->weights_3[i];
    }
}

void resetNNUE(NNUE* nnue) {
    for (int i = 0; i < INPUTS_COUNT; ++i) {
        nnue->inputs[i] = 0;
    }

    for (int i = 0; i < INNER_LAYER_COUNT; ++i) {
        nnue->accumulators[i] = 0;
    }

    nnue->eval = 0;
}


void modifyNnue(NNUE* nnue, Board* board, int color, int piece) {
    for(int sq = 0; sq < 64; ++sq) {
        int weight = isExists(board, color, piece, sq);
        if (weight) {
            setNNUEInput(nnue, getInputIndexOf(color, piece, sq));
        } else {
            resetNNUEInput(nnue, getInputIndexOf(color, piece, sq));
        }
    }
}

void loadNNUEWeights() {
    FILE* file = fopen("./fc1.weights.csv", "r");
    if (file == NULL) {
        perror("Unable to open file");
        exit(1);
    }

    for (int i = 0; i < INNER_LAYER_COUNT; i++) {
        for (int j = 0; j < INPUTS_COUNT; j++) {
            if (fscanf(file, "%lf,", &nnue->weights_1[i][j]) != 1) {
                perror("Error reading file");
                fclose(file);
                exit(1);
            }
        }
    }
    fclose(file);

    file = fopen("./fc2.weights.csv", "r");

    if (file == NULL) {
        perror("Unable to open file");
        exit(1);
    }

    for (int i = 0; i < INNER_LAYER_COUNT; i++) {
      for (int j = 0; j < INNER_LAYER_COUNT; j++) {
          if (fscanf(file, "%lf,", &nnue->weights_2[i][j]) != 1) {
              perror("Error reading file");
              fclose(file);
              exit(1);
          }
      }
    }

    fclose(file);

    file = fopen("./fc3.weights.csv", "r");

    if (file == NULL) {
        perror("Unable to open file");
        exit(1);
    }

    for (int i = 0; i < INNER_LAYER_COUNT; i++) {
        if (fscanf(file, "%lf,", &nnue->weights_3[i]) != 1) {
            perror("Error reading file");
            fclose(file);
            exit(1);
        }
    }

    fclose(file);

    resetNNUE(nnue);
}

//void initNNUEWeights() {
//    // load file with weights
//}

void initNNUEPosition(NNUE* nnue, Board* board) {
    resetNNUE(nnue);

    modifyNnue(nnue, board, WHITE, PAWN);
    modifyNnue(nnue, board, BLACK, PAWN);
    modifyNnue(nnue, board, WHITE, KNIGHT);
    modifyNnue(nnue, board, BLACK, KNIGHT);
    modifyNnue(nnue, board, WHITE, BISHOP);
    modifyNnue(nnue, board, BLACK, BISHOP);
    modifyNnue(nnue, board, WHITE, ROOK);
    modifyNnue(nnue, board, BLACK, ROOK);
    modifyNnue(nnue, board, WHITE, QUEEN);
    modifyNnue(nnue, board, BLACK, QUEEN);
    modifyNnue(nnue, board, WHITE, KING);
    modifyNnue(nnue, board, BLACK, KING);
}

TimeManager createFixNodesTm(int nodes) {
    TimeManager tm = initTM();
    tm.nodes = nodes;
    tm.depth = 100;
    tm.searchType = FixedNodes;
    return tm;
}

int MAX_FEN_LENGTH = 1000;

void dataset_gen(Board* board, int from, int to, char* filename) {
    TimeManager tm = createFixNodesTm(10000);
    FILE *inputFile = fopen("ccrl_positions.txt", "r");
    FILE *outputFile = fopen(filename, "w");

    if (inputFile == NULL || outputFile == NULL) {
        perror("Ошибка открытия файла");
        exit(1);
    }

    char fen[MAX_FEN_LENGTH];
    int lineNumber = 0;

    while (fgets(fen, MAX_FEN_LENGTH, inputFile) != NULL) {
        lineNumber++;

        if (lineNumber < from) {
            continue;
        }

        if (lineNumber >= to) {
            fclose(inputFile);
            fclose(outputFile);
            exit(0);
            return;
        }

        char *newline = strchr(fen, '\n');
        if (newline) {
            *newline = '\0';
        }

        // Обрабатываем FEN-позицию
        printf("Позиция #%d: %s\n", lineNumber, fen);

        setFen(board, fen);
        SearchInfo info = iterativeDeeping(board, tm);
        int absoluteEval = board->color == WHITE ? info.eval : -info.eval;
        printf("Eval: %d\n", absoluteEval);
        fprintf(outputFile, "%s,%d\n", fen, absoluteEval);
    }


//    setFen(board, "rnbqkbnr/8/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
//    SearchInfo info = iterativeDeeping(board, tm);
//    int absoluteEval = board->color == WHITE ? info.eval : -info.eval;
//    printf("Eval: %d\n", absoluteEval);

    fclose(inputFile);
    fclose(outputFile);
    exit(0);
}