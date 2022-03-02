from typing import List

class Individual:
    def __init__(self, genes: List[str]) -> None:
        self.genes = genes
        self.gold = 0
        self.fitness = 0
        self.path = []
