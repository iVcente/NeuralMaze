class Neuron:
    def __init__(self, n_weights: int, role: str) -> None:
        self.input_values = []
        self.n_weights = n_weights
        self.role = role
        self.weights = []
        self.y_value = 0
