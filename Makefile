CC = gcc
CFLAGS = -std=c11 -m64
SRC = *.c
OPTIMIZATIONS = -O3 -march=native -flto
WARNINGS = -Wall -pedantic
DEBUG = -g
OUTPUT = zevra

all:
	$(CC) $(CFLAGS) $(OPTIMIZATIONS) $(WARNINGS) $(SRC) -o $(OUTPUT)
debug:
	$(CC) $(CFLAGS) $(DEBUG) $(WARNINGS) $(SRC) -o $(OUTPUT)

clean:
	rm -rf *.o $(OUTPUT)