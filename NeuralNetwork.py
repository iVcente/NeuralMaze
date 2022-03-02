from typing import List

from Neuron import Neuron

class NeuralNetwork:
    def __init__(self, hidden_layer: List[Neuron], output_layer: List[Neuron]) -> None:
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
