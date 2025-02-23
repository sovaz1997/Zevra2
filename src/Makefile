CC = gcc
CFLAGS = -std=gnu17 -m64 -fcommon
SRC = *.c
OPTIMIZATIONS = -O3 -flto
NATIVE = -march=native
WARNINGS = -Wall -Wextra -Wshadow
DEBUG = -g -Wall -pedantic -fno-omit-frame-pointer -gdwarf-2
LIBS = -lpthread -lm
OUTPUT = zevra
OUTPUT_X86 = zevra_x86
OUTPUT_ARM = zevra_arm

ARCH_X86 = -mavx2
ARCH_ARM = -mfpu=neon

ARCH = $(shell uname -m)

ifeq ($(ARCH), x86_64)
    ARCH_FLAGS = $(ARCH_X86)
    OUTPUT_BINARY = $(OUTPUT_X86)
else ifeq ($(ARCH), armv7l)
    ARCH_FLAGS = $(ARCH_ARM)
    OUTPUT_BINARY = $(OUTPUT_ARM)
else
    $(error Unsupported architecture: $(ARCH))
endif

all:
	$(CC) $(CFLAGS) -DNDEBUG $(OPTIMIZATIONS) $(NATIVE) $(ARCH_FLAGS) $(SRC) -o $(OUTPUT_BINARY) $(LIBS)

debug:
	$(CC) $(CFLAGS) $(DEBUG) $(WARNINGS) $(SRC) -o $(OUTPUT_BINARY) $(LIBS)

release: all

clean:
	rm -rf *.o $(OUTPUT_BINARY)
