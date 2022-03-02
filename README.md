python3 NeuralMaze.py -f maze.txt -p 500 -g 500 -c 0.5 -m 0.5 -e fast

O arquivo a ser executado se chama *NeuralMaze.py*. O programa possui os seguintes parâmetros obrigatórios: 
- f: Arquivo a ser lido. Uma string contendo extensão do arquivo
- p: População inicial. Um número inteiro maior que zero
- g: Número de gerações que o algoritmo deve executar. Um número inteiro maior que zero
- c: Taxa de realização de crossover. Um número de ponto flutuante entre zero e um
- m: Taxa de mutação. Um número de ponto flutuante entre zero e um
- e: Modo de execução. "fast" ou "slow"

Exemplos de execução:
`python3 NeuralMaze.py -f maze.txt -p 750 -g 500 -c 0.5 -m 0.1 -e fast`
Mais informações podem ser obtidas com o seguinte comando:
`python3 NeuralMaze.py --help`
