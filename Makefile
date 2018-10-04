CC = gcc
SRC = *.c
OPTIMIZATIONS = -O3 -march=native -flto
WARNINGS = -Wall -pedantic
OUTPUT = zevra

all:
	$(CC) $(OPTIMIZATIONS) $(WARNINGS) $(SRC) -o $(OUTPUT)

clean:
	rm -rf *.o $(OUTPUT)