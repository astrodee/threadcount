class Error(Exception):
    """Base class of other error classes."""
    def __init__(self, message):
        Exception.__init__(self)
        self.message = message

    def __str__(self):
        return f"{self.message}"

class EmptyRegionError(Error):
    def __init__(self, message="No region is selected."):
        super().__init__(message)

class InputDimError(Error):
    def __init__(self, dim, message="The input data must be 3-d."):
        self.dim = dim
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'The array is {self.dim}-d. {self.message}'

class InputShapeError(Error):
    def __init__(self, message=""):
        super().__init__(message)

