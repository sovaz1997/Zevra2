# Zevra 2

![Logo](https://i.ibb.co/rd64j8q/Zevra-Logo-Horizontal.png)

Zevra 2 is a free and open-source UCI chess engine, written in C and licensed
under the GPLv3.

Zevra has used an NNUE evaluation since 2.6. Version **2.7** overhauls that
network and largely rewrites the search, for roughly **+400 Elo** over 2.6 in
internal testing.

## What's new in 2.7

**Evaluation and data.**

- The network is trained from scratch on ~1.5B positions of Zevra's own
  self-play. That data comes from a reinforcement loop seeded by the 2.6
  network: each generation's champion generates games, a fresh network is
  trained from scratch on them, and the stronger net then generates the next,
  better games.
- Fixed a data-generation bug that had corrupted the training labels; retraining
  on the clean data was worth about +170 Elo on its own.
- A non-linear **int8 evaluation head** with a VNNI fast path (and an AVX2
  fallback, so int8 inference is fast even without AVX-VNNI): about +82.
- A **factorised king-buckets** feature transformer (8 buckets) with an
  incremental int8 accumulator on the engine side.
- Network shape: `(768 -> 256) x2 -> 32 -> 1`, SCReLU activations. The network
  (`zevra.bin`, ~3 MB) ships with the engine and is loaded at startup; it can be
  swapped via the `EvalFile` UCI option or the `ZEVRA_NET` environment variable.

**Search.**

- SPSA-tuned search parameters (LMR base/divisor, futility / reverse-futility /
  razor margins, null-move reduction, aspiration) plus Late Move Pruning: +57.
- Continuation (counter-move) history, 1+2 ply, bonus/malus with gravity: +12.7.
- Context-aware late move reductions (PV / killer / history / improving) and a
  second full SPSA round that rebalanced null-move and futility: +14.
- Adaptive time management — soft/hard limits, best-move stability, fail-low
  panic: +49.5 on a real clock.
- Fixed three AddressSanitizer/UBSan bugs (killer indexing, stage-index
  overflow on promotions, a negative shift in pawn attacks): +5.

NNUE is always enabled in 2.7 — there is no longer an evaluation toggle.

## Strength

Zevra 2.7 is single-threaded. In the author's own blitz gauntlets at 8+0.08 it
is on par with Andscacs 0.94 and Stockfish 7 and clearly above Critter 1.6a and
Hakkapeliitta 3.0, which puts it roughly in the 3100–3150 range. These are
self-testing figures, not an official rating-list result.

## Downloads / which build to use

Release binaries are provided per CPU instruction-set level. Pick the highest
one your CPU supports (they are otherwise identical in playing strength):

| Build         | CPU                                                   |
|---------------|-------------------------------------------------------|
| `x86-64`      | Any 64-bit x86 — universal fallback (slower)           |
| `avx2`        | Haswell (2013) and newer Intel, AMD Zen 1+            |
| `avx-vnni`    | Intel Alder Lake and newer, AMD Zen 4+               |
| `avx512-vnni` | Cascade Lake / Ice Lake servers, AMD Zen 4           |

Keep `zevra.bin` next to the executable (or point `EvalFile` at it).

If unsure, `avx2` runs on virtually every CPU from 2013 on and is within a few
percent of the faster builds thanks to the AVX2 int8 fallback. The `x86-64`
build runs on any 64-bit CPU but is noticeably slower (scalar evaluation) — use
it only if `avx2` reports an illegal-instruction error.

## Building from source

Requires `gcc` (or `clang`) and the bundled `src/zevra.bin`. From `src/`:

```
make            # build optimized for the current machine (-march=native)
make tiers      # build all three distributable tiers (static, LTO)
make avx2       # build a single tier (also: make avx-vnni / make avx512-vnni)
```

- On **Windows**, build under MSYS2. The **MINGW64** environment links against
  `msvcrt.dll` and produces binaries that run on any Windows (7 and up, no
  runtime to install); UCRT64 also works but needs Windows 10+. gcc appends
  `.exe` to the output name.
- On **macOS / Apple Silicon**, use plain `make` (macOS cannot link `-static`,
  so the tier targets do not apply). arm64 currently uses the scalar evaluation
  path (no NEON kernels yet), so it runs but is slower than on x86.

## UCI options

| Option        | Description                                                     |
|---------------|-----------------------------------------------------------------|
| `Hash`        | Transposition table size in MB.                                 |
| `Clear Hash`  | Clears the transposition table.                                 |
| `EvalFile`    | Path to the NNUE network file (default `zevra.bin`).            |
| `Temperature` | Move-selection randomness, 0–100 (0 = strongest/deterministic). |

## The neural network

The evaluation network is trained from scratch with a private [bullet](https://github.com/jw1912/bullet)-based
trainer, entirely on positions from Zevra's own self-play — no external engine's
data was used. The training data comes from a reinforcement loop seeded by the
2.6 network (self-play → train a fresh net from scratch → stronger net → more
self-play). Only the trained network (`zevra.bin`) ships with the engine.

## Project inspired by

- Chess Programming Wiki: https://www.chessprogramming.org/Main_Page
- Stockfish: https://github.com/official-stockfish/Stockfish
- Ethereal: https://github.com/AndyGrant/Ethereal
- bullet (NNUE trainer): https://github.com/jw1912/bullet

## Thanks

Guardian, Graham Banks (the creator of many tournaments with different engines),
Ratosh (Pirarucu dev.), Daniel Anulliero (Isa dev.), Evgeny Kotlov (Hedgehog
dev.), Sergey Kudryavtsev (sdchess.ru creator), Rasmus Althoff (CT800 author).

## License

GPLv3 — see [LICENSE](LICENSE). Copyright (C) 2018 Oleg Smirnov.
