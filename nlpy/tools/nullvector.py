import numpy as np


class NullVectorIterator(object):

  def __init__(self, start, step, stop, value):
    self._start = start
    self._stop = stop
    self._step = step
    self._current = start
    self._value = value

  def __iter__(self):
    return self

  def next(self):
    next = self._current + self._step
    if (self._step > 0 and next < self._stop) or \
       (self._step < 0 and next > self._stop):
      self._current = next
      return self._value
    raise StopIteration

  def __next__(self):
    return self.next()


class NullVector(object):
  """
  An object that simulates a vector with all
  components equal to a given value.
  """

  def __init__(self, length, value=0, dtype=np.float):
    self._dtype = dtype if isinstance(dtype, type) else dtype.type
    self._value = self.dtype(value)
    self._length = length

  @property
  def value(self):
      return self._value

  @property
  def dtype(self):
      return self._dtype

  @property
  def shape(self):
    return (len(self),)

  def full(self):
    "Convert NullVector to Numpy array."
    return self.value * np.ones(len(self), dtype=self.dtype)

  def __len__(self):
    return self._length

  def __getitem__(self, i):
    if isinstance(i, int):
      l = len(self)
      if i >= -l and i < l:
        return self.value
      raise IndexError("Index out of bounds.")

    if hasattr(i, 'len'):
      return NullVector(len(i), self.value, self.dtype)

    if isinstance(i, slice):
      return NullVector(len(xrange(*i.indices(len(self)))),
                        self.value, self.dtype)

    raise TypeError('Invalid index type.')

  def __contains__(self, other):
    return other == self.value

  def __iter__(self):
    return NullVectorIterator(0, 1, len(self), self.value)

  def __reversed__(self):
    return self.__iter__()

  def __add__(self, other):
    if np.isscalar(other):
      return NullVector(len(self),
                        self.value + other,
                        dtype=np.result_type(self.dtype, other))

    if isinstance(other, np.ndarray):
      if len(other) == len(self):
        x = other.copy()
        x += self.value
        return x
      raise ValueError("Adding vectors of different size.")

    if isinstance(other, NullVector):
      if len(other) == len(self):
        result_type = np.result_type(self.dtype, other.dtype)
        return NullVector(len(self),
                          self.value + other.value,
                          dtype=result_type)
      raise ValueError("Adding vectors of different size.")

    return NotImplemented

  def __radd__(self, other):
    return self.__add__(other)

  def __iadd__(self, other):
    if np.isscalar(other):
      self._value += other
      return self

    if isinstance(other, NullVector):
      if len(other) == len(self):
        self._value += other.value
      raise ValueError("Adding vectors of different size.")

    return NotImplemented

  def __sub__(self, other):
    return self.__add__(-other)

  def __rsub__(self, other):
    return self.__sub__(other)

  def __isub__(self, other):
    return self.__iadd__(-other)

  def __mul__(self, other):
    if np.isscalar(other):
      return NullVector(len(self),
                        self.value * other,
                        dtype=np.result_type(self.dtype, other))

    if isinstance(other, np.ndarray):
      x = other.copy()
      x *= self.value
      return x

    if isinstance(other, NullVector):
      if len(other) == len(self):
        return NullVector(len(self),
                          self.value * other.value,
                          dtype=np.result_type(self.dtype, other.dtype))
      raise ValueError("Multiplying vectors of different size.")

    return NotImplemented

  def __rmul__(self, other):
    return self.__mul__(other)

  def __imul__(self, other):
    if np.isscalar(other):
      self._value *= other
      return self

    if isinstance(other, NullVector):
      if len(other) == len(self):
        self._value *= other.value
      raise ValueError("Multiplying vectors of different size.")

    return NotImplemented

  def __div__(self, other):
    if np.isscalar(other):
      return NullVector(len(self),
                        self.value / other,
                        dtype=np.result_type(self.dtype, other))

    if isinstance(other, NullVector):
      if len(other) == len(self):
        return NullVector(len(self),
                          self.value / other.value,
                          dtype=np.result_type(self.dtype, other.dtype))
      raise ValueError("Dividing vectors of different size.")

    return NotImplemented

  def __rdiv__(self, other):
    return NotImplemented

  def __idiv__(self, other):
    if np.isscalar(other):
      self._value /= other
      return self

    if isinstance(other, NullVector):
      if len(other) == len(self):
        self._value /= other.value
      raise ValueError("Multiplying vectors of different size.")

    return NotImplemented

  def __truediv__(self, other):
    return self.__div__(other)

  def __rtruediv__(self, other):
    return self.__rdiv__(other)

  def __pow__(self, other):
    if np.isscalar(other):
      return NullVector(len(self),
                        self.value ** other,
                        dtype=np.result_type(self.dtype, other))

    return NotImplemented

  def __ipow__(self, other):
    if np.isscalar(other):
      self._value **= other
      return self

  def __neg__(self):
    return NullVector(len(self), -self.value, self.dtype)

  def __pos__(self):
    return self

  def __abs__(self):
    return NullVector(len(self), abs(self.value), self.dtype)

  def __eq__(self, other):
    if np.isscalar(other):
      return range(len(self)) if self.value == other else []

    if isinstance(other, NullVector):
      if len(self) == len(other) and self.value == other.value:
        return range(len(self))
      return []

    if isinstance(other, np.ndarray):
      if len(other) == len(self):
        return np.where(other == self.value)[0]
      return []

    return NotImplemented

  def __lt__(self, other):
    if np.isscalar(other):
      return range(len(self)) if self.value < other else []

    if isinstance(other, NullVector):
      if len(self) == len(other) and self.value < other.value:
        return range(len(self))
      return []

    if isinstance(other, np.ndarray):
      if len(other) == len(self):
        return np.where(other < self.value)[0]
      return []

    return NotImplemented

  def __gt__(self, other):
    if np.isscalar(other):
      return range(len(self)) if self.value > other else []

    if isinstance(other, NullVector):
      if len(self) == len(other) and self.value > other.value:
        return range(len(self))
      return []

    if isinstance(other, np.ndarray):
      if len(other) == len(self):
        return np.where(other > self.value)[0]
      return []

    return NotImplemented

  def __le__(self, other):
    if np.isscalar(other):
      return range(len(self)) if self.value <= other else []

    if isinstance(other, NullVector):
      if len(self) == len(other) and self.value <= other.value:
        return range(len(self))
      return []

    if isinstance(other, np.ndarray):
      if len(other) == len(self):
        return np.where(other <= self.value)[0]
      return []

    return NotImplemented

  def __ge__(self, other):
    if np.isscalar(other):
      return range(len(self)) if self.value >= other else []

    if isinstance(other, NullVector):
      if len(self) == len(other) and self.value >= other.value:
        return range(len(self))
      return []

    if isinstance(other, np.ndarray):
      if len(other) == len(self):
        return np.where(other >= self.value)[0]
      return []

    return NotImplemented

  def __complex__(self):
    return NullVector(len(self), self.value, dtype=np.complex)

  def __int__(self):
    return NullVector(len(self), int(self.value), dtype=np.int)

  def __long__(self):
    return NullVector(len(self), long(self.value), dtype=np.int)

  def __float__(self):
    return NullVector(len(self), float(self.value), dtype=np.float)

  def __coerce__(self, other):
    if isinstance(other, NullVector):
      result_type = np.result_type(self.dtype, other.dtype)
      x = NullVector(len(self), self.value, dtype=result_type)
      y = NullVector(len(other), other.value, dtype=result_type)
      return (x, y)

    return NotImplemented

  def __repr__(self):
    v = "%g" % self.value
    l = len(self)
    t = str(self.dtype)
    return "NullVector([%s ... %s], %s) of size %d" % (v, v, t, l)
