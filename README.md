Zevra 2 - полностью переписанный шахматный uci-движок.

Преимущества по сравнению с Zevra v1.8.6:

+ Perft-герератор в 2 раза быстрее;
+ Уменьшен размер ячейки хеш-таблицы;
+ Ускорена инициализация движка;
+ Размер кода < 3000 строк;
+ Уход от ООП-подхода;
+ Реализация SEE.

На данный момент Zevra 2 находится в beta-версии и обгоняет Zevra v1.6.1 на~100
пунктов ЭЛО в контроле 60s+0.6s (~2380-2400 ЭЛО CCRL 40/4).

Результаты тестирования:
```
Zevra v2.0 beta vs. Zevra v1.8.6:
Score: 200 - 87 - 134  [0.634] 421
Elo difference: 95.60 +/- 27.96
```

Проект вдохновлен:
+ Chess programming wiki: https://www.chessprogramming.org/Main_Page
+ Stockfish: https://github.com/official-stockfish/Stockfish
+ Ethereal: https://github.com/AndyGrant/Ethereal

Thanks: Guardian, Graham Banks, Ratosh (Pirarucu developer), Daniel Anulliero (Isa developer), Евгений Котлов (Hedhehog author).