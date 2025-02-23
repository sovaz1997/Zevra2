CC = gcc
CFLAGS = -std=gnu17 -m64 -fcommon
SRC = *.c
OPTIMIZATIONS = -O3 -flto
NATIVE = -march=native
WARNINGS = -Wall -Wextra -Wshadow
DEBUG = -g -Wall -pedantic -fno-omit-frame-pointer -gdwarf-2
LIBS = -lpthread -lm
OUTPUT = zevra

ARCH_X86 = -mavx2
ARCH_ARM = -mfpu=neon

ARCH = $(shell uname -m)


ifeq ($(ARCH), x86_64)
    ARCH_FLAGS = $(ARCH_X86)
else ifeq ($(ARCH), arm64)
    ARCH_FLAGS = $(ARCH_ARM)
else
    $(error Unsupported architecture: $(ARCH))
endif

all:
	$(CC) $(CFLAGS) -DNDEBUG $(OPTIMIZATIONS) $(NATIVE) $(ARCH_FLAGS) $(SRC) -o $(OUTPUT) $(LIBS)

debug:
	$(CC) $(CFLAGS) $(DEBUG) $(WARNINGS) $(SRC) -o $(OUTPUT) $(LIBS)

release: all

clean:
	rm -rf *.o $(OUTPUT)
