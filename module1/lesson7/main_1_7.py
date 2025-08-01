from circle import Circle
from parses.json_handler import JsonHandler
from parses.secure_text_handler import SecureTextHandler
from rectangle import Rectangle
from triangle import Triangle
from vector import Vector

if __name__ == '__main__':
    tr_1 = Triangle(3, 2, 4)
    rc_1 = Rectangle(5, 9)
    cr_1 = Circle(5)
    square_1 = tr_1.area()
    square_2 = cr_1.area()
    square_3 = rc_1.area()
    vector_1 = Vector([3, 3, -5])
    vector_3 = Vector([3, 3, -5])
    vector_2 = Vector([1, 7, 4])

    r = {'cookies': {},
         'body': 'hello'
         }

    print(SecureTextHandler(request=r).process())
