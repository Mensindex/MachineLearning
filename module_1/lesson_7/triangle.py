from module_1.lesson_7.base_figure import BaseFigure


class Triangle(BaseFigure):
    a: int
    b: int
    c: int
    n_dots = 3

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        super().__init__()

    def validate(self):
        if self.a > self.b + self.c or self.b > self.a + self.c or self.c > self.a + self.b:
            raise ValueError("triangle inequality does not hold")
        else:
            return self.a, self.b, self.c

    def area(self):
        p = (self.a + self.b + self.c) / 2
        return (p * (p - self.a) * (p - self.b) * (p - self.c)) ** 0.5
