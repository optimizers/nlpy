class List(list):
    """
    The List type extends the default list type to allow list shifting by a
    scalar. Examples:

        >>> x = List([1,2,3])  # List with a capital L
        >>> x
        [1, 2, 3]
        >>> x + [4,5,6]        # Usual list concatenation
        [1, 2, 3, 4, 5, 6]
        >>> x + 2              # Shift all elements of x by 2
        [3, 4, 5]
        >>> x                  # x was not altered
        [1, 2, 3]
        >>> x += 2             # Now x is altered
        >>> x
        [3, 4, 5]
        >>> x * 2              # Usual list multiplication
        [3, 4, 5, 3, 4, 5]

    Subtraction works similarly, both in-place and 'out of place'.

    No other operations on lists are subclassed. Therefore, multiplication
    by an integer still returns concatenated copies of the list.
    """

    def __add__(self, other):
        # Allows the operation  list = list + scalar
        if isinstance(other, int) or isinstance(other, float):
            new = self[:]
            for i in xrange(len(new)):
                new[i] += other
            return List(new)
        else:
            return List(list.__add__(self, other))

    def __radd__(self, other):
        # Allows the operation  list = scalar + list
        return self.__add__(other)

    def __iadd__(self, other):
        # Allows the operation  list += scalar
        if isinstance(other, int) or isinstance(other, float):
            for i in xrange(len(self)):
                self[i] += other
            return self
        else:
            self += other
            return self

    def __sub__(self, other):
        # Allows the operation  list = list - scalar
        if isinstance(other, int) or isinstance(other, float):
            return self.__add__(-other)
        else:
            return List(list.__sub__(self, other))

    def __rsub__(self, other):
        # Allows the operation  list = scalar - list
        if isinstance(other, int) or isinstance(other, float):
            new = self[:]
            for i in xrange(len(new)):
                new[i] = other - new[i]
            return List(new)
        else:
            return List(list.__rsub__(self,other))

    def __isub__(self, other):
        # Allows the operation  list -= scalar
        if isinstance(other, int) or isinstance(other, float):
            for i in xrange(len(self)):
                self[i] -= other
            return self
        else:
            self -= other
            return self

def _test(): 
    import doctest
    return doctest.testmod()
    
if __name__ == "__main__": 
    _test() 
