CC = gcc
CFLAGS = -std=gnu17 -m64
SRC = *.c
OPTIMIZATIONS = -O3 -flto
NATIVE = -march=native
WARNINGS = -Wall -Wextra -Wshadow
DEBUG = -g -Wall -pedantic -fno-omit-frame-pointer -gdwarf-2
LIBS = -lpthread -lm
OUTPUT = zevra

ARCH_X86 = -mavx2
# AArch64 (Apple Silicon) has NEON as a baseline, so no arch flag is needed. The
# NNUE currently has no NEON intrinsics, so arm64 uses the scalar fallback paths
# (correct but slower). (-mfpu=neon is a 32-bit-ARM GCC flag and is rejected by
# clang on Apple Silicon.)
ARCH_ARM =

# Instruction-set tiers for distribution. All share the int8 king-buckets net
# (src/zevra.bin); pick the highest your CPU supports. The int8 matmul has an
# AVX2 fallback, so avx2 is already within ~8% of the VNNI builds.
#   avx2        : Haswell (2013)+, Zen 1+         (AVX2 + FMA + POPCNT)
#   avx-vnni    : Alder Lake+, Zen 4+             (+ AVX-VNNI)
#   avx512-vnni : Cascade Lake+ / Ice Lake servers, Zen 4 (+ AVX-512 VNNI/VL)
#   x86-64      : any 64-bit x86 (baseline SSE2, scalar NNUE — slow but universal)
TIER_BASE   = -march=x86-64
TIER_AVX2   = -mavx2 -mfma -mpopcnt
TIER_VNNI   = -mavx2 -mfma -mpopcnt -mavxvnni
TIER_AVX512 = -mavx2 -mfma -mpopcnt -mavx512vnni -mavx512vl

# Champion network architecture: king-buckets + int8 head (needs a matching
# KB net, e.g. src/zevra.bin). Override with `make NET_ARCH=` for a plain
# Chess768 engine (to load old 256/512 nets).
NET_ARCH = -DNNUE_KB=8 -DNNUE_L2_I8=32

ARCH = $(shell uname -m)


ifeq ($(ARCH), x86_64)
    ARCH_FLAGS = $(ARCH_X86)
else ifeq ($(ARCH), arm64)
    ARCH_FLAGS = $(ARCH_ARM)
else
    $(error Unsupported architecture: $(ARCH))
endif

all:
	$(CC) $(CFLAGS) -DNDEBUG $(OPTIMIZATIONS) $(NATIVE) $(ARCH_FLAGS) $(NET_ARCH) $(EXTRA) $(SRC) -o $(OUTPUT) $(LIBS)

debug:
	$(CC) $(CFLAGS) $(DEBUG) $(WARNINGS) $(NET_ARCH) $(EXTRA) $(SRC) -o $(OUTPUT) $(LIBS)

# Tuning build: exposes the SPSA search/TM parameters as UCI options (hidden in
# normal builds). Use for local SPSA runs, not for release.
tune:
	$(CC) $(CFLAGS) -DNDEBUG $(OPTIMIZATIONS) $(NATIVE) $(ARCH_FLAGS) $(NET_ARCH) -DTUNING_ENABLED=1 $(EXTRA) $(SRC) -o $(OUTPUT) $(LIBS)

release:
	$(CC) $(CFLAGS) -DNDEBUG $(OPTIMIZATIONS) -static $(ARCH_FLAGS) $(NET_ARCH) $(SRC) -o $(OUTPUT) $(LIBS)

# Distribution tiers: statically linked, per instruction-set level. Output is
# suffixed ($(OUTPUT)-avx2 etc.); on Windows/MSYS2 gcc appends .exe. `make tiers`
# builds all three.
REL = $(CC) $(CFLAGS) -DNDEBUG $(OPTIMIZATIONS) -static $(NET_ARCH) $(EXTRA)

x86-64:
	$(REL) $(TIER_BASE) $(SRC) -o $(OUTPUT)-x86-64 $(LIBS)

avx2:
	$(REL) $(TIER_AVX2) $(SRC) -o $(OUTPUT)-avx2 $(LIBS)

avx-vnni:
	$(REL) $(TIER_VNNI) $(SRC) -o $(OUTPUT)-avx-vnni $(LIBS)

avx512-vnni:
	$(REL) $(TIER_AVX512) $(SRC) -o $(OUTPUT)-avx512-vnni $(LIBS)

tiers: x86-64 avx2 avx-vnni avx512-vnni

clean:
	rm -rf *.o $(OUTPUT) $(OUTPUT)-x86-64 $(OUTPUT)-avx2 $(OUTPUT)-avx-vnni $(OUTPUT)-avx512-vnni
