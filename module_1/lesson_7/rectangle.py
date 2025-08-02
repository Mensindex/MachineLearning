from base_figure import BaseFigure


class Rectangle(BaseFigure):
    n_dots = 4

    def __init__(self, a, b):
        self.a = a
        self.b = b
        super().__init__()

    def validate(self):
        return self.a, self.b

    def area(self):
        return self.a * self.b
