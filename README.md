# Zevra 2

![Logo](https://i.ibb.co/rd64j8q/Zevra-Logo-Horizontal.png)

Zevra 2 is free and open-source UCI chess engine.

## Compilation commands

+ `make all` - creating a build optimized for your CPU (will not work on other PCs)
+ `make popcnt` - creating a universal build with support for POPCNT instructions
+ `make nonpopcnt` - creating a universal build without the support of the POPCNT instruction
+ `make release` - creating a universal builds with POPCNT and without POPCNT instruction

## Game level

### Last versions of Zevra

```
# PLAYER                      :  RATING  ERROR  POINTS  PLAYED   (%)
1 WyldChess                   :  2678.1   22.4   813.0    1072    76
2 Zevra v2.4 r380             :  2589.6   21.1   697.0    1083    64
3 Zevra v2.3 r348 popcnt      :  2543.0   20.8   617.5    1073    58
4 Galjoen 0.41.2              :  2503.7   20.7   553.5    1071    52
5 CT800 V1.43 64 bit          :  2493.8   20.7   540.0    1074    50
6 Zevra v2.2.1 r328 popcnt    :  2433.7   20.6   441.5    1070    41
7 Loki 3.5.0                  :  2421.5   20.9   424.5    1074    40
8 Zevra v2.1.2 r248           :  2419.0   ----   419.0    1071    39
9 Teki 2                      :  2355.1   21.3   324.0    1072    30
```

## Project inspired
+ Chess programming wiki: https://www.chessprogramming.org/Main_Page
+ Stockfish: https://github.com/official-stockfish/Stockfish
+ Ethereal: https://github.com/AndyGrant/Ethereal

## Thanks
Guardian, Graham Banks (the creator of many tournaments with
different engines), Ratosh (Pirarucu dev.), Daniel Anulliero(Isa dev.),
Evgeny Kotlov (Hedhehog dev.), Sergey Kudryavtsev (sdchess.ru creator), Rasmus Althoff (CT800 author).
