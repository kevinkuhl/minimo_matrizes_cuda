# Valor mínimo de duas matrizes

Programa em CUDA que utiliza os conceitos de operações atômicas e também de múltiplas streams para calcular o menor valor presente em duas matrizes inseridas através de um arquivo

Exemplo de arquivo de matrizes:

10 3
2 3
1 2 3
4 5 6
7 8 9
10 11 12
13 14 15
16 17 18
-1 -2 -3
-4 -5 -6
-7 -8 -9
-10 -11 -12
5 -10
9 8 7
6 5 4

Onde as duas primeiras linhas representam as dimensões da primeira e segunda matriz respectivamente.

Para executar o programa, primeiro

\ $ make 

Em seguida

\ $ ./calc_matriz <nomeDaEntrada.txt>
