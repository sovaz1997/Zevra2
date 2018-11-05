CC = gcc
CFLAGS = -std=c11 -m64
SRC = *.c
OPTIMIZATIONS = -O3 -flto
NATIVE = -march=native
WARNINGS = -Wall -pedantic
popcnt = -msee4.2 -mpopcnt
DEBUG = -g -Wall -pedantic -fno-omit-frame-pointer -gdwarf-2
LIBS = -lpthread -lm
OUTPUT = zevra.exe
OUTPUT_POPCNT = zevra_popcnt.exe
OUTPUT_NONPOPCNT = zevra_nonpopcnt.exe


all:
	$(CC) $(CFLAGS) $(OPTIMIZATIONS) $(NATIVE) $(SRC) -o $(OUTPUT) $(LIBS)
popcnt:
	$(CC) $(CFLAGS) $(OPTIMIZATIONS) $(POPCNT) $(SRC) -o $(OUTPUT_POPCNT) $(LIBS)
nonpopcnt:
	$(CC) $(CFLAGS) $(OPTIMIZATIONS) $(SRC) -o $(OUTPUT_NONPOPCNT) $(LIBS)
debug:
	$(CC) $(CFLAGS) $(DEBUG) $(WARNINGS) $(SRC) -o $(OUTPUT) $(LIBS)

clean:
	rm -rf *.o $(OUTPUT)