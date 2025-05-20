class SizeError(Exception):
    def __init__(self, message: str=""):
        super().__init__(message)


class ShapeError(Exception):
    def __init__(self, message: str=""):
        super().__init__(message)


class GeometryError(Exception):
    def __init__(self, message: str=""):
        super().__init__(message)
