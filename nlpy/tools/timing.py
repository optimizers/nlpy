# Platform-dependent time measurement.

try:
    # Use resource module if available.
    import resource
    def cputime():
        return resource.getrusage(resource.RUSAGE_SELF)[0]
except:
    # Fall back on time.clock().
    import time
    def cputime():
        return time.clock()
