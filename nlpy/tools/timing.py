# $Id:$
import resource

def cputime():
	return resource.getrusage(resource.RUSAGE_SELF)[0]
