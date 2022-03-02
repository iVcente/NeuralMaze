from typing import List

class Maze:
    def __init__(self, structure: List[List[str]]) -> None:
        self.structure = structure
        self.exit_pos = [len(structure) - 1, len(structure[0]) - 1]
        self.previous_pos = [0, 0]
        self.current_pos = [0, 0] # Starting point
        self.directions = { # Possible directions
            "N": [-1, 0], # North
            "S": [1, 0],  # South
            "W": [0, -1], # West
            "E": [0, 1]   # East
        }

    def current_element(self) -> str:
        return self.structure[self.current_pos[0]][self.current_pos[1]]

    def reset_position(self) -> None:
        self.current_pos = [0, 0]

    def get_element_score(self, element: str) -> int:
        if (element == "Free cell"):
            return 10
        if (element == "Off limits"):
            return -50
        if (element == "Wall"):
            return -10
        if (element == "Gold"):
            return 50
        if (element == "Entrance"):
            return -10
        else: # element == "Exit"
            return 100

    def check_next_pos(self, next_direction: str) -> List[int]:
        next_row = self.current_pos[0] + self.directions[next_direction][0]
        next_col = self.current_pos[1] + self.directions[next_direction][1]

        return [next_row, next_col]

    def traverse(self, direction: str) -> str:
        next_row = self.current_pos[0] + self.directions[direction][0]
        next_col = self.current_pos[1] + self.directions[direction][1]
        self.previous_pos = self.current_pos
        self.current_pos = [next_row, next_col]

        # Off limits
        if (next_row < 0 or next_col < 0):
            return "Off limits"
        if (next_row >= len(self.structure) or next_col >= len(self.structure[0])):
            return "Off limits"

        # Valid position
        curr_elem = self.current_element()
        if (curr_elem == '0'):
            return "Free cell"
        if (curr_elem == '1'):
            return "Wall"
        if (curr_elem == 'M'):
            return "Gold"
        if (curr_elem == 'E'):
            return "Entrance"
        else: # curr_elem == 'S
            return "Exit"

    def get_neighbors(self) -> List[int]:
        neighbors = []
        for direction in self.directions.values():
            row = self.current_pos[0] +  direction[0]
            col = self.current_pos[1] +  direction[1]

            # Off limits
            if (row < 0 or col < 0):
                neighbors.append(-1)
                continue
            if (row >= len(self.structure) or col >= len(self.structure[0])):
                neighbors.append(-1)
                continue

            # Valid position
            neighbor = self.__check_neighbor(direction)
            neighbors.append(neighbor)

        return neighbors

    def __check_neighbor(self, direction: List[int]) -> int:
        row = self.current_pos[0] +  direction[0]
        col = self.current_pos[1] +  direction[1]

        # Valid position
        neighbor = self.structure[row][col]

        if (neighbor == '0' or neighbor == 'E'): # Free cell or starting point
            return 0
        elif (neighbor == '1'): # Wall
            return 1
        elif (neighbor == 'M'): # Gold
            return 2
        else: # Exit
            return 3
