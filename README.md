# Zevra 2

Zevra 2 - uci chess engine.

Advantages:

+ Perft-gen 2-3x-faster;
+ Initialization speedup;
+ Code length < 3000;
+ +100 elo on Zevra v1.8.6.

It's ~2350-2450 elo on CCRL now.

# Test results
```
tc=60s+0.6s
Hash=64mb

Score of Zevra v2.0 r172 vs Zevra v1.8.6 r672 popcnt: 251 - 110 - 208  [0.624] 569
Elo difference: 87.93 +/- 23.03
```

Tests vs. other engines: soon!

# Project inspired
+ Chess programming wiki: https://www.chessprogramming.org/Main_Page
+ Stockfish: https://github.com/official-stockfish/Stockfish
+ Ethereal: https://github.com/AndyGrant/Ethereal

# Thanks
Guardian, Graham Banks (the creator of many tournaments with
different engines), Ratosh (Pirarucu dev.), Daniel Anulliero(Isa dev.),
Евгений Котлов (Hedhehog dev.), Сергей Кудрявцев (sdchess.ru creator).