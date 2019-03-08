# Zevra 2

![Logo](https://s8.hostingkartinok.com/uploads/images/2018/11/4294efcd52c48d08915a9a2fc643f978.png)

Zevra 2 is free and open-source UCI chess engine.

## Compilation commands

+ `make all` - creating a version optimized for your CPU (will not work on other PCs)
+ `make popcnt` - creating a universal assembly with support for POPCNT instructions
+ `make nonpopcnt` - creating a universal assembly without the support of the POPCNT instruction
+ `make release` - creating a universal assembly with POPCNT and without POPCNT instruction

## Game level

### Zevra v2.1.1 r216

+ 2468 Elo in [SCET 60s+0.6s](https://sites.google.com/view/scet-testing/zevra)
+ 2380 Elo in [CCRL 40/40](http://ccrl.chessdom.com/ccrl/4040/cgi/engine_details.cgi?print=Details&each_game=1&eng=Zevra%202.1.1%20r216%2064-bit#Zevra_2_1_1_r216_64-bit)


## Project inspired
+ Chess programming wiki: https://www.chessprogramming.org/Main_Page
+ Stockfish: https://github.com/official-stockfish/Stockfish
+ Ethereal: https://github.com/AndyGrant/Ethereal

## Thanks
Guardian, Graham Banks (the creator of many tournaments with
different engines), Ratosh (Pirarucu dev.), Daniel Anulliero(Isa dev.),
Евгений Котлов (Hedhehog dev.), Сергей Кудрявцев (sdchess.ru creator).