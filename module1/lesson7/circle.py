import math

from base_figure import BaseFigure


class Circle(BaseFigure):
    n_dots = float('inf')
    r: int

    def __init__(self, r: int):
        self.r = r
        super().__init__()

    def validate(self):
        pass

    def area(self):
        return math.pi * self.r * 2
