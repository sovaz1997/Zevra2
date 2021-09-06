#include "board.h"

typedef struct TuningPosition TuningPosition;
typedef struct CalculateArgs CalculateArgs;

double fun(Board* board);
TuningPosition* makeCopyPositions();
void makeTuning(Board* board);
int* getValues();
void* calculate(void* args);
void setValues(int* values);
void changeParam(int n, int value);
void printParams();
void printPST(char* name, int* pst, int* curIndex, FILE* f);
void printArray(char* name, int* arr, int* curIndex, int length, FILE* f);
int* transferPST(int* from, int* to, int* curIndex);
int* transfer(int* from, int* to, int* curIndex, int length);
void loadPositions(Board* board);
char** str_split(char* a_str, const char a_delim);