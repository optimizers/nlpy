#!/usr/bin/env python
# 
# A sparse vector class for Python
# D. Orban, 2004
#
import numpy
import types
import operator
import math

"""
A dictionary-based sparse vector class
which supports elementwise mathematical operations
and operations with other sparse vectors
and numpy arrays.

$Id: sparse_vector_class.py 88 2008-09-29 04:43:15Z d-orban $
"""

class SparseVector:
    """
    A dictionary-based sparse vector class. To initialize
    a sparse vector in R^1000, use, e.g.,
        v = SparseVector( 1000, { 1 : 0.1, 10: 0.01, 100 : 0.001 } )
    v contains only 3 nonzero elements, at (0-based) indices
    1, 10 and 100 with respective values 0.1, 0.01 and 0.001.
    To initialize an empty sparse vector, use
        v = SparseVector( 1000, {} )
    """

    def __init__( self, n, d0 ):
        if not isinstance( n, types.IntType ):
            raise TypeError, "Vector size must be integer"
        if n <= 0:
            raise ValueError, "Vector size must be positive"
        if not isinstance( d0, types.DictType ):
            raise TypeError, "Vector contents must be a dictionary"

        self.n = n
        self.values = {}
        for k in d0.keys():
            self.values[k] = d0[k]

    # Insert a new item in sparse vector
    # If index is larger than vector size, adjust vector size
    def __setitem__( self, index, item ):
        if not isinstance( index, types.IntType ):
            raise KeyError, "Index must be integer"
        if index > self.n: self.n = index
        if not operator.isNumberType( item ):
            raise TypeError, "Value must be numeric"
        if float( item ) != 0.0:
            dict.__setitem__( self.values, index, item )

    # Fetch item from sparse vector
    def __getitem__( self, index ):
        if not isinstance( index, types.IntType ):
            raise KeyError, "Index must be integer"
        if index >= self.n:
            raise IndexError, "Index out of range"
        if index in self.keys():
            return dict.__getitem__(self.values,index)
        else:
            return 0

    # Obtain segment of sparse vector --- treat vector as circular
    def __getslice__( self, i, j ):
        if not isinstance( i, types.IntType ) or not isinstance( j, types.IntType ):
            raise KeyError, "Indices must be integer"
        slice = {}
        K = self.keys()
        K.sort()
        if i <= j:
            size = j-i
            for k in K:
                if i <= k and k < j:
                    slice[k] = self.values[k]
        else:
            size = self.n
            for k in K:
                if j > k or k >= i:
                    slice[k] = self.values[k]
        return SparseVector( size, slice )
        
    def __add__( self, other ):
        # Order of testing is important!
        if isinstance( other, numpy.ndarray ):
            # If adding sparse and dense, result is dense
            #rv = SparseVector( max( self.n, other.shape[0] ), {} )
            rv = numpy.zeros( max( self.n, other.shape[0] ), 'd' )
            for k in range( other.shape[0] ): rv[k] = other[k]
            for k in self.values.keys():
                rv[k] += self[k]
            return rv
        elif isSparseVector( other ):
            rv = SparseVector( max( self.n, other.n ), {} )
            for k in self.values.keys():
                rv[k] += self[k]
            for k in other.values.keys():
                rv[k] += other[k]
            return rv
        elif operator.isNumberType( other ):
            rv = SparseVector( self.n, {} )
            for k in self.values.keys():
                rv[k] = self[k] + other
            return rv
        else:
            raise TypeError, "Cannot add with SparseVector"

    def __radd__( self, other ):
        return SparseVector.__add__( self, other )

    def __iadd__( self, other ):
        # Order of testing is important!
        if isinstance( other, numpy.ndarray ):
            self.n = max( self.n, other.shape[0] )
            for k in range( other.shape[0] ):
                self[k] += other[k]
            return self
        elif isSparseVector( other ):
            self.n = max( self.n, other.n )
            inter = filter( self.values.has_key, other.values.keys() )
            other_only = filter( lambda x: not self.values.has_key, other.values.keys() )
            for k in inter + other_only:
                self[k] += other[k]
            return self
        elif operator.isNumberType( other ):
            for k in self.values.keys():
                self[k] += other
            return self
        else:
            raise TypeError, "Cannot add to SparseVector"

    def __isub__( self, other ):
        return SparseVector.__iadd__( self, -other )

    def __imul__( self, other ):
        """
        Element by element multiplication. For dot product,
        see the helper function dot().
        """
        if isinstance( other, numpy.ndarray ):
            self.n = max( self.n, other.shape[0] )
            for k in range( other.shape[0] ):
                self.values[k] *= other[k]
            return self
        elif isSparseVector( other ):
            self.n = max( self.n, other.n )
            inter = filter( self.values.has_key, other.values.keys() )
            other_only = filter( lambda x: not self.values.has_key, other.values.keys() )
            for k in inter + other_only:
                self.values[k] *= other.values[k]
            return self
        elif operator.isNumberType( other ):
            for k in self.values.keys():
                self.values[k] *= other
            return self
        else:
            raise TypeError, "Cannot multiply with SparseVector"

    def __idiv__( self ):
        """
        Element by element division.
        """
        if isinstance( other, numpy.ndarray ):
            self.n = max( self.n, other.shape[0] )
            for k in range( other.shape[0] ):
                self[k] /= other[k]
            return self
        elif isSparseVector( other ):
            self.n = max( self.n, other.n )
            inter = filter( self.values.has_key, other.values.keys() )
            other_only = filter( lambda x: not self.values.has_key, other.values.keys() )
            for k in inter + other_only:
                self[k] /= other.values[k]
            return self
        elif operator.isNumberType( other ):
            for k in self.values.keys():
                self[k] /= other
            return self
        else:
            raise TypeError, "Cannot multiply with SparseVector"

    def __neg__( self ):
        rv = SparseVector( self.n, {} )
        for k in self.values.keys(): rv[k] = -self[k]
        return rv
    
    def __sub__( self, other ):
        return SparseVector.__add__( self, -other )

    def __rsub__( self, other ):
        return SparseVector.__sub__( self, other )

    def __mul__( self, other ):
        """
        Element by element multiplication. For dot product,
        see the helper function dot().
        """
        if isinstance( other, numpy.ndarray ):
            rv = SparseVector( max( self.n, other.shape[0] ), {} )
            for k in self.values.keys():
                rv[k] = self[k] * other[k]
            return rv
        elif isSparseVector( other ):
            rv = SparseVector( max( self.n, other.n ), {} )
            for k in filter( self.values.has_key, other.values.keys() ):
                rv[k] = self[k] * other[k]
            return rv
        elif operator.isNumberType( other ):
            rv = SparseVector( self.n, {} )
            for k in self.values.keys():
                rv[k] = self[k] * other
            return rv
        else:
            raise TypeError, "Cannot multiply with SparseVector"

    def __rmul__(self, other):
        return SparseVector.__mul__( self, other )

    def __div__(self, other):
        """
        Element by element division.
        """
        if isinstance( other, numpy.ndarray ):
            rv = SparseVector( max( self.n, other.shape[0] ), {} )
            for k in self.values.keys():
                rv[k] = self[k] / other[k]
            return rv
        elif isSparseVector( other ):
            rv = SparseVector( max( self.n, other.n ), {} )
            for k in filter( self.values.has_key, other.values.keys() ):
                rv[k] = self[k] / other[k]
            return rv
        elif operator.isNumberType( other ):
            rv = SparseVector( self.n, {} )
            for k in self.values.keys():
                rv[k] = self[k] / other
            return rv
        else:
            raise TypeError, "Cannot multiply with SparseVector"

    def __rdiv__(self, other):
        """
        The same as __div__
        """
        return SparseVector.__div__( self, other )

    def __pow__( self, other ):
        """
        Raise each element of sparse vector to a power.
        If power is another sparse vector, compute elementwise power.
        In this latter case, by convention, 0^0 = 0.
        """
        if not isSparseVector( self ):
            raise TypeError, "Argument must be a SparseVector"
        if isSparseVector( other ):
            rv = SparseVector( max( self.n, other.n ), {} )
            for k in self.values.keys():
                rv[k] = self[k]**other[k]
            return rv
        if not isinstance( other, types.IntType )  and \
           not isinstance( other, types.LongType ) and \
           not isinstance( other, types.FloatType ):
                raise TypeError, "Power must be numeric or a sparse vector"
        rv = SparseVector( self.n, {} )
        for k in self.values.keys():
            rv[k] = math.pow( self[k], other )
        return rv
        
    def __rpow__( self, other ):
        """
        Use each element of sparse vector as power of base
        """
        if not isSparseVector( self ):
            raise TypeError, "Argument must be a SparseVector"
        if not isinstance( other, types.IntType )  and \
           not isinstance( other, types.LongType ) and \
           not isinstance( other, types.FloatType ):
                raise TypeError, "Power must be numeric"
        rv = SparseVector( self.n, {} )
        for k in self.values.keys():
            rv[k] = math.pow( other, self[k] )
        return rv

    def __repr__( self ):
        s = 'SparseVector( %-d, {' % self.n
        for k in self.values.keys():
            s += '%-d' % k
            s += ' : %-g, ' % self.values[k]
        s += '} )'
        return s

    def __str__( self ):
        nnz = self.nnz()
        s = ' Sparse Vector of size %-d, %-d nonzeros\n' % (self.size(), nnz)
        K = self.values.keys()
        K.sort()
        s += ' Values:\n'
        for k in K:
            s += '  %-5d\t%-g\n' % (k, self.values[k])
        return s

    ###########################################################################

    def keys( self ):
        return self.values.keys()

    def size( self ):
        "Return vector size"
        return self.n

    def nnz( self ):
        "Return number of nonzero elements"
        return len( self.values )

    def resize( self, m ):
        """
        Adjust vector size. If m is larger than current size,
        reset current size to m. Otherwise, reset current size
        to m and drop each value whose index is beyond m.
        """
        if m < self.n:
            # Drop all elements beyon m
            for k in self.keys():
                if k >= m:
                    self.values.__delitem__( k )
        else:
            self.n = m

    def shrink( self ):
        "Shrink vector size to largest index"
        self.n = max( self.values.keys() )

    def to_list( self ):
        "Convert sparse vector to (dense) list"
        rv = []
        for k in range( self.n ):
            if self.values.has_key( k ):
                rv.append( self.values[k] )
            else:
                rv.append( 0.0 )
        return rv

    def to_array( self ):
        "Convert sparse vector to (dense) numpy array"
        try:
            import numpy
        except:
            raise ImportError, "Unable to import module numpy"
            return None
        rv = numpy.zeros( self.n, 'd' )
        for k in self.values.keys():
            rv[k] = self.values[k]
        return rv

    def out( self ):
        print self
        return

###############################################################################


def isSparseVector( x ):
    """
    Determines if the argument is a SparseVector object.
    """
    return hasattr(x,'__class__') and x.__class__ is SparseVector

def zeros( n ):
    """
    Returns a zero vector of length n.
    """
    return SparseVector( n, {} )

def ones( n, indlist = None ):
    """
    Returns a vector of length n with all ones in the
    specified positions (default: range(n)).
    """
    if indlist is None: indlist = range( n )
    rv = SparseVector( n, {} )
    for k in indlist: rv.values[k] = 1
    return rv

def random(n, lmin = 0.0, lmax = 1.0, indlist = None):
    """
    Returns a sparse vector of length n with random
    values in the range [lmin,lmax] in the specified
    positions (default: max(5,n/100) random positions).
    """
    import whrandom
    import random
    gen = whrandom.whrandom()
    dl = lmax-lmin
    rv = SparseVector( n, {} )
    if indlist is None:
        nval = max( 5, n/100 )
        indlist = random.sample( xrange( n ), nval )
    for k in indlist: rv.values[k] = dl * gen.random()
    return rv

def dotss( a, b ):
    """
    dot product of two sparse vectors
    """
    dotproduct = 0.
    for k in filter( a.values.has_key, b.values.keys() ):
        dotproduct += a[k] * b[k]
    return dotproduct

def dotsn( a, b ):
    """
    dot product of a sparse vector and numpy array
    """
    dotproduct = 0.
    for k in a.keys():
        dotproduct += a[k] * b[k]
    return dotproduct
    
def dot(a, b):
    """
    dot product of two vectors.
    """
    if isSparseVector(a) and isSparseVector(b):
        return dotss( a, b )
    elif isSparseVector(a) and isinstance( b, numpy.ndarray ):
        return dotsn( a, b )
    elif isinstance( a, numpy.ndarray ) and isSparseVector( b ):
        return dotsn( b, a )
    elif isinstance( a, numpy.ndarray ) and isinstance( b, numpy.ndarray ):
        return numpy.dot( a, b )
    else:
        raise TypeError, "Arguments must be SparseVector and/or numpy array"
    return dotproduct
    

def norm(a):
    """
    Computes the 2-norm of vector a.
    """
    if not isSparseVector( a ):
        raise TypeError, "Argument must be a SparseVector"
    return math.sqrt(dot(a,a))

def norm2(a):
    """
    Computes the 2-norm of vector a.
    """
    return norm(a)

def norm1(a):
    """
    Computes the 1-norm of vector a.
    """
    return sum( [ abs( a.values[k] ) for k in a.values.keys() ] )

def norm_infty(a):
    """
    Computes the infinity-norm of vector a.
    """
    return max( [ abs( a.values[k] ) for k in a.values.keys() ] )

def normp(a,p):
    """
    Computes the p-norm of vector a.
    """
    if p <= 0:
        raise ValueError, "p must be positive"
    if p == 1: return norm1(a)
    if p == 2: return norm2(a)
    return sum( [ abs( a.values[k] )**p for k in a.values.keys() ] )**(1.0/p)

def sum(a):
    """
    Returns the sum of the elements of a.
    """
    if not isSparseVector( a ):
        raise TypeError, "Argument must be a SparseVector"
    return sum( [a.values[k] for k in a.values.keys()] )

# elementwise operations
    
def log10(a):
    """
    log10 of each element of a.
    """
    if not isSparseVector( a ):
        raise TypeError, "Argument must be a SparseVector"
    rv = SparseVector( a.n, {} )
    for k in a.values.keys():
        rv.values[k] = math.log10( a.values[k] )
    return rv

def log(a):
    """
    log of each element of a.
    """
    if not isSparseVector( a ):
        raise TypeError, "Argument must be a SparseVector"
    rv = SparseVector( a.n, {} )
    for k in a.values.keys():
        rv.values[k] = math.log( a.values[k] )
    return rv
        
def exp(a):
    """
    Elementwise exponential.
    """
    if not isSparseVector( a ):
        raise TypeError, "Argument must be a SparseVector"
    rv = SparseVector( a.n, {} )
    for k in a.values.keys():
        rv.values[k] = math.exp( a.values[k] )
    return rv

def sin(a):
    """
    Elementwise sine.
    """
    if not isSparseVector( a ):
        raise TypeError, "Argument must be a SparseVector"
    rv = SparseVector( a.n, {} )
    for k in a.values.keys():
        rv.values[k] = math.sin( a.values[k] )
    return rv
        
def tan(a):
    """
    Elementwise tangent.
    """
    if not isSparseVector( a ):
        raise TypeError, "Argument must be a SparseVector"
    rv = SparseVector( a.n, {} )
    for k in a.values.keys():
        rv.values[k] = math.tan( a.values[k] )
    return rv
        
def cos(a):
    """
    Elementwise cosine.
    """
    if not isSparseVector( a ):
        raise TypeError, "Argument must be a SparseVector"
    rv = SparseVector( a.n, {} )
    for k in a.values.keys():
        rv.values[k] = math.cos( a.values[k] )
    return rv

def asin(a):
    """
    Elementwise inverse sine.
    """
    if not isSparseVector( a ):
        raise TypeError, "Argument must be a SparseVector"
    rv = SparseVector( a.n, {} )
    for k in a.values.keys():
        rv.values[k] = math.asin( a.values[k] )
    return rv

def atan(a):
    """
    Elementwise inverse tangent.
    """ 
    if not isSparseVector( a ):
        raise TypeError, "Argument must be a SparseVector"
    rv = SparseVector( a.n, {} )
    for k in a.values.keys():
        rv.values[k] = math.atan( a.values[k] )
    return rv

def acos(a):
    """
    Elementwise inverse cosine.
    """
    if not isSparseVector( a ):
        raise TypeError, "Argument must be a SparseVector"
    rv = SparseVector( a.n, {} )
    for k in a.values.keys():
        rv.values[k] = math.acos( a.values[k] )
    return rv

def sqrt(a):
    """
    Elementwise sqrt.
    """
    if not isSparseVector( a ):
        raise TypeError, "Argument must be a SparseVector"
    rv = SparseVector( a.n, {} )
    for k in a.values.keys():
        rv.values[k] = math.sqrt( a.values[k] )
    return rv

def sinh(a):
    """
    Elementwise hyperbolic sine.
    """
    if not isSparseVector( a ):
        raise TypeError, "Argument must be a SparseVector"
    rv = SparseVector( a.n, {} )
    for k in a.values.keys():
        rv.values[k] = math.sinh( a.values[k] )
    return rv

def tanh(a):
    """
    Elementwise hyperbolic tangent.
    """
    if not isSparseVector( a ):
        raise TypeError, "Argument must be a SparseVector"
    rv = SparseVector( a.n, {} )
    for k in a.values.keys():
        rv.values[k] = math.tanh( a.values[k] )
    return rv

def cosh(a):
    """
    Elementwise hyperbolic cosine.
    """
    if not isSparseVector( a ):
        raise TypeError, "Argument must be a SparseVector"
    rv = SparseVector( a.n, {} )
    for k in a.values.keys():
        rv.values[k] = math.cosh( a.values[k] )
    return rv

def atan2(a,b):    
    """
    Arc tangent of a/b
    """
    if not isSparseVector( a ):
        raise TypeError, "Argument must be a SparseVector"
    rv = SparseVector( a.n, {} )
    for k in a.values.keys():
        rv.values[k] = math.atan2( a.values[k], b )
    return rv
    
###############################################################################
#
##   Perform series of basic tests
#
###############################################################################

if __name__ == "__main__":

    print 'a = zeros(4)'
    a = zeros(4)

    print 'a.__doc__=',a.__doc__

    print 'a[0] = 1.0'
    a[0] = 1.0

    print 'a[3] = 3.0'
    a[3] = 3.0

    print 'a[0]=', a[0]
    print 'a[1]=', a[1]

    print 'a.size()=', a.size()
            
    b = SparseVector( 4, {0:1, 1:2, 2:3, 3:4} )
    print 'a=', a
    print 'b=', b

    print 'a+b'
    c = a + b
    c.out()

    print '-a'
    c = -a
    c.out()

    print 'a-b'
    c = a - b
    c.out()

    print 'a*1.2'
    c = a*1.2
    c.out()


    print '1.2*a'
    c = 1.2*a
    c.out()

    print 'dot(a,b) = ', dot(a,b)
    print 'dot(b,a) = ', dot(b,a)

    print 'a*b'
    c = a*b
    c.out()
    
    print 'a/1.2'
    c = a/1.2
    c.out()

    print 'a[0:2]'
    c = a[0:2]
    c.out()

    #print 'a[2:5] = [9.0, 4.0, 5.0]'
    #a[2:5] = [9.0, 4.0, 5.0]
    #a.out()

    print 'sqrt(a)=',sqrt(a)
    print 'pow(a, 2*ones( a.size() ) )=',pow(a, 2*ones( a.size() ) )
    print 'pow(a, 2)=',pow(a, 2)

    print 'ones(10)'
    c = ones(10)
    c.out()

    print 'zeros(10)'
    c = zeros(10)
    c.out() 

    print 'del a'
    del a

    print 'a = random( 11, 0., 2.)'
    try:
        a = random(11, 0., 2.)
        a.out()

    except:
        print '   failed!'
        pass

