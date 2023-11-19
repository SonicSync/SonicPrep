from dataclasses import dataclass


@dataclass
class Variation:
    def __init__(self, root, name, data):
        self.root = root
        self.name = name
        self.data = data
