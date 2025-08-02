class Vector:
    coords: list[int]

    def __init__(self, coords: list[int]):
        self.coords = coords

    def __add__(self, other):
        if len(self.coords) != len(other.coords):
            raise ValueError(f"left and right lengths differ: {len(self.coords)}!={len(other.coords)}")
        else:
            result = list()
            for i in range(len(self.coords)):
                result.append(self.coords[i] + other.coords[i])
            return Vector(coords=result)

    def __mul__(self, other):
        if isinstance(other, int):
            for i in range(len(self.coords)):
                self.coords[i] *= other
            return self
        elif isinstance(other, Vector):
            if len(self.coords) != len(other.coords):
                raise ValueError(f"left and right lengths differ: {len(self.coords)}!={len(other.coords)}")
            else:
                result = 0
                for i in range(len(self.coords)):
                    result += self.coords[i] * other.coords[i]
                return result
        else:
            raise ValueError(f"object {other} is wrong type")

    def __abs__(self):
        squares_sum = 0
        for coord in self.coords:
            squares_sum += coord * coord
        return squares_sum ** 0.5

    def __eq__(self, other):
        if isinstance(other, Vector):
            return self.coords == other.coords
        else:
            raise ValueError(f"object {other} is not Vector type")

    def __str__(self):
        return f"{self.coords}"
