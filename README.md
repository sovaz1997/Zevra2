# Zevra 2

Zevra 2 is free and open-source UCI chess engine.

## Compilation commands

+ `make all` - creating a version optimized for your CPU (will not work on other PCs)
+ `make popcnt` - creating a universal assembly with support for POPCNT instructions
+ `make nonpopcnt` - creating a universal assembly without the support of the POPCNT instruction

## Strength of the game

### Zevra v2.0 r172

[CCRL 40/4](http://www.computerchess.org.uk/ccrl/404/cgi/engine_details.cgi?print=Details&each_game=1&eng=Zevra%202.0%20r172%2064-bit#Zevra_2_0_r172_64-bit): 2408 elo points

[SCET 60s+0.6s](https://sites.google.com/view/scet-testing/zevra)


## Regression tests (dev-versions)
#### Time control
```
tc: 10s+0.1s
Hash: 16mb
```

#### 12.11.2018
```
Score of Zevra v2.1 r193 dev vs Zevra v2.0 r172: 767 - 491 - 742  [0.569] 2000
Elo difference: 48.25 +/- 12.11
```
#### 11.11.2018
```
Score of Zevra v2.1 r183 dev vs Zevra v2.0 r172: 355 - 254 - 391  [0.550] 1000
Elo difference: 35.21 +/- 16.82
```

![Regression graph](https://s8.hostingkartinok.com/uploads/images/2018/11/137707b2216cb5e8758dabbfdce20b69.png)

## Project inspired
+ Chess programming wiki: https://www.chessprogramming.org/Main_Page
+ Stockfish: https://github.com/official-stockfish/Stockfish
+ Ethereal: https://github.com/AndyGrant/Ethereal

## Thanks
Guardian, Graham Banks (the creator of many tournaments with
different engines), Ratosh (Pirarucu dev.), Daniel Anulliero(Isa dev.),
Евгений Котлов (Hedhehog dev.), Сергей Кудрявцев (sdchess.ru creator).