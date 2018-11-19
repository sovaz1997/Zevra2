# Zevra 2

![Logo](https://s8.hostingkartinok.com/uploads/images/2018/11/4294efcd52c48d08915a9a2fc643f978.png)

Zevra 2 is free and open-source UCI chess engine.

## Compilation commands

+ `make all` - creating a version optimized for your CPU (will not work on other PCs)
+ `make popcnt` - creating a universal assembly with support for POPCNT instructions
+ `make nonpopcnt` - creating a universal assembly without the support of the POPCNT instruction
+ `make release` - creating a universal assembly with POPCNT and without POPCNT instruction

## Strength of the game

### Zevra v2.1 r210

+ 2468 Elo in [SCET 60s+0.6s](https://sites.google.com/view/scet-testing/zevra)

### Zevra v2.0 r172

+ [CCRL 40/4](http://www.computerchess.org.uk/ccrl/404/cgi/engine_details.cgi?print=Details&each_game=1&eng=Zevra%202.0%20r172%2064-bit#Zevra_2_0_r172_64-bit): 2408 elo points
+ 2429 Elo in [SCET 60s+0.6s](https://sites.google.com/view/scet-testing/zevra)


## Regression tests (dev-versions)
#### Time control
```
tc: 10s+0.1s
Hash: 16mb
```

#### 19.11.2018
```
Score of Zevra v2.2 r227 dev vs Zevra v2.1.1 r216: 1920 - 1399 - 1681  [0.552] 5000
Elo difference: 36.33 +/- 7.86
```

![Regression graph](https://s8.hostingkartinok.com/uploads/images/2018/11/dea76a4e813452134e406433bf0ceb60.png)

## Project inspired
+ Chess programming wiki: https://www.chessprogramming.org/Main_Page
+ Stockfish: https://github.com/official-stockfish/Stockfish
+ Ethereal: https://github.com/AndyGrant/Ethereal

## Thanks
Guardian, Graham Banks (the creator of many tournaments with
different engines), Ratosh (Pirarucu dev.), Daniel Anulliero(Isa dev.),
Евгений Котлов (Hedhehog dev.), Сергей Кудрявцев (sdchess.ru creator).