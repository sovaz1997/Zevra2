CC = gcc
CFLAGS = -std=gnu11 -m64
SRC = *.c
OPTIMIZATIONS = -O3 -flto
NATIVE = -march=native
WARNINGS = -Wall -Wextra -Wshadow
POPCNT = -DUSE_POPCNT -msse3 -mpopcnt
DEBUG = -g -Wall -pedantic -fno-omit-frame-pointer -gdwarf-2
LIBS = -lpthread -lm
OUTPUT = zevra
OUTPUT_POPCNT = zevra_popcnt
OUTPUT_NONPOPCNT = zevra_nonpopcnt

all:
	$(CC) $(CFLAGS) -DNDEBUG $(OPTIMIZATIONS) $(NATIVE) $(SRC) -o $(OUTPUT) $(LIBS)
popcnt:
	$(CC) $(CFLAGS) -DNDEBUG $(OPTIMIZATIONS) -static $(POPCNT) $(SRC) -o $(OUTPUT_POPCNT) $(LIBS)
nonpopcnt:
	$(CC) $(CFLAGS) -DNDEBUG $(OPTIMIZATIONS) -static $(SRC) -o $(OUTPUT_NONPOPCNT) $(LIBS)
debug:
	$(CC) $(CFLAGS) $(DEBUG) $(WARNINGS) $(SRC) -o $(OUTPUT) $(LIBS)
release:
	make popcnt
	make nonpopcnt

clean:
	rm -rf *.o $(OUTPUT)