class BaseFigure:
    n_dots = None

    def __init__(self):
        self.validate()

    def area(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError
