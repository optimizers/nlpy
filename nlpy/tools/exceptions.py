# NLPy-specific exceptions.

# Use this exception, e.g., in a post iteration to exit based
# on a custom stopping condition.
class UserExitRequest(Exception):
    """
    Exception that the caller can use to request clean exit.
    """
    def __init__(self):
        pass


class InfeasibleError(Exception):
    """
    Error that can be raised to signal an infeasible iterate.
    """
    def __init__(self):
    pass
