How to run:
```
python3 NeuralMaze.py -f maze.txt -p 750 -g 500 -c 0.5 -m 0.1 -e fast
```

More details on the mandatory parameters:
- _f_: File to be read. A string containing the file name and its extension;
- _p_: Initial population. An integer greater than zero;
- _g_: Maximum number of generations. An integer greater than zero;
- _c_: Crossover rate. A floating point number between zero and one;
- _m_: Mutation rate. A floating point number between zero and one;
- _e_: Execution mode. Two options: "fast" or "slow".

More info can be obtained with:
```
python3 NeuralMaze.py --help
```
