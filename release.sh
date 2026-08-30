#!/bin/bash
# Build the release binaries for every instruction-set tier on the current
# platform, name them <version>-<platform>-<tier>, and drop the network file
# next to them under dist/.
#
#   Linux : bash release.sh
#   Windows (MSYS2 UCRT64): CC=/c/msys64/ucrt64/bin/gcc.exe bash release.sh
#   macOS : bash release.sh            (single non-static build; tiers are x86)
set -e

VERSION=2.7
NET=zevra.bin
CC=${CC:-gcc}

ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/src"
OUT="$ROOT/dist"

case "$(uname -s)" in
    Linux)                PLAT=linux;   EXT=;     STATIC=-static ;;
    MINGW*|MSYS*|CYGWIN*) PLAT=windows; EXT=.exe; STATIC=-static ;;
    Darwin)               PLAT=macos;   EXT=;     STATIC=       ;;  # macOS can't link -static
    *)                    PLAT=$(uname -s); EXT=; STATIC=-static ;;
esac

CFLAGS="-std=gnu17 -m64 -DNDEBUG -O3 -flto -s"   # -s: strip symbols from the release binary
NET_ARCH="-DNNUE_KB=8 -DNNUE_L2_I8=32"
LIBS="-lpthread -lm"

# tier name -> instruction flags
TIER_NAMES="x86-64 avx2 avx-vnni avx512-vnni"
tier_flags() {
    case "$1" in
        x86-64)      echo "-march=x86-64" ;;   # baseline SSE2, runs on any 64-bit x86 (scalar NNUE)
        avx2)        echo "-mavx2 -mfma -mpopcnt" ;;
        avx-vnni)    echo "-mavx2 -mfma -mpopcnt -mavxvnni" ;;
        avx512-vnni) echo "-mavx2 -mfma -mpopcnt -mavx512vnni -mavx512vl" ;;
    esac
}

mkdir -p "$OUT"
cp "$SRC/$NET" "$OUT/$NET"
: > "$OUT/build.log"
echo "Platform: $PLAT   CC: $CC"

cd "$SRC"
for tier in $TIER_NAMES; do
    bin="zevra-$VERSION-$PLAT-$tier$EXT"
    echo ">> $bin"
    # Compiler warnings (harmless %llu/U64 format notes) go to build.log.
    $CC $CFLAGS $STATIC $(tier_flags "$tier") $NET_ARCH *.c -o "$OUT/$bin" $LIBS 2>>"$OUT/build.log"
done

echo
echo "Built into $OUT:"
ls -la "$OUT"
echo
echo "Sanity bench (each tier should print 1524503 nodes):"
for tier in $TIER_NAMES; do
    bin="$OUT/zevra-$VERSION-$PLAT-$tier$EXT"
    printf '  %-28s ' "$(basename "$bin")"
    ( cd "$OUT" && printf 'bench 13\nquit\n' | "$bin" 2>/dev/null | grep -oE '[0-9]+ nodes' | head -1 )
done
