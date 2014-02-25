# NLPy-specific exceptions.


class UserExitRequest(Exception):
    """
    Exception that the caller can use to request clean exit.
    """
    def __init__(self):
        pass


class EqualityConstraintsError(Exception):
    """
    Exception that signals a problem with equality constraints.
    """
    def __init__(self):
        pass


class InequalityConstraintsError(Exception):
    """
    Exception that signals a problem with inequality constraints.
    """
    def __init__(self):
        pass


class BoundConstraintsError(Exception):
    """
    Exception that signals a problem with bound constraints.
    """
    def __init__(self):
        pass


class GeneralConstraintsError(Exception):
    """
    Exception that signals a problem with general constraints.
    """
    def __init__(self):
        pass


class InfeasibleError(Exception):
    """
    Error that can be raised to signal an infeasible iterate.
    """
    def __init__(self):
        pass


class ShapeError(Exception):
    """
    Error that can be raised to signal a dimension mismatch.
    """
    def __init__(self):
        pass
