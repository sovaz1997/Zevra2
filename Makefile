CC = gcc
CFLAGS = -std=c11 -m64
SRC = *.c
OPTIMIZATIONS = -O3 -march=native -flto
WARNINGS = -Wall -pedantic
DEBUG = -g -Wall -pedantic -fno-omit-frame-pointer -gdwarf-2
LIBS = -lpthread -lm
OUTPUT = zevra.exe

all:
	$(CC) $(CFLAGS) $(OPTIMIZATIONS) $(SRC) -o $(OUTPUT) $(LIBS)
debug:
	$(CC) $(CFLAGS) $(DEBUG) $(WARNINGS) $(SRC) -o $(OUTPUT) $(LIBS)

clean:
	rm -rf *.o $(OUTPUT)