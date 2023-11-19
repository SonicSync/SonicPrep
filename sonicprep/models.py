from dataclasses import dataclass


@dataclass
class Variation:
    def __init__(self, root, name):
        self.root = root
        self.name = name
