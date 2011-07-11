#!/usr/bin/env python
#
# pprof.py, v0.5: Create performance profiles.
# Written by Michael P. Friedlander <michael@mcs.anl.gov>
# Updated by Dominique Orban for use with matplotlib
# <Dominique.Orban@polymtl.ca>
#
import getopt, sys, re
import numpy
import numpy.ma as ma
from   string import atoi, atof

PROGNAME = "nlpy_pprof.py"

def usage():
    instructions = """
Usage: %s [OPTION]... [FILE 1] [FILE 2]...[FILE N]
Create a performance profile chart from metrics in FILEs.
Output is an eps file sent to stdout.

Example 1:
  Profile the metrics in column 3 of the files solver1, solver2, and solver3.
  Use a log2 scale for the x-axis.  Redirect the stdout to profile.eps.

    %s -l 2 -c 3 solver1 solver2 solver3 > profile.eps

Example 2:
  Specify a title, linestyle and failure threshold.  Pop up an X window.

    %s -c 3 -t "Plot title" --linestyle "linespoints" \\
             --term "x11" solver1 solver2 solver3

See Dolan and More',
   "Benchmarking optimization software with performance profiles",
   available at http://www-unix.mcs.anl.gov/~more/cops/

Options
  -c, --column=COLUMN      get metrics from column COLUMN (default 1)
      --cpu                specify that CPU times are being compared
  -h, --help               get some help
  -l, --logscale=BASE      logscaleBASE scale for x-axis (default linear)
      --legend             insert a legend
      --linestyle=STYLE    use STYLE as Gnuplot line style (default steps)
      --sep RE             use regexp RE to indicate new column (default space)
      --backend=TERM       use TERM backend (default TkAgg)
      --thresh=eps         only compare data values larger than eps >= 0. Useful
                           when used in conjunction with --cpu.
  -t, --title=LABEL        use LABEL as  title  (default none)
  -x, --xlabel=LABEL       use LABEL for x-axis (default none)
  -y, --ylabel=LABEL       use LABEL for y-axis (default none)

- Use non-positive values to indicate that the algorithm failed.
- Use --sep 'S' to indicate columns are separated by character S.
- Use --sep "r'RE'" to separate instead by a regular expression.
- Any line starting with a %% or a # is ignored.
""" % (PROGNAME,PROGNAME,PROGNAME)

    print instructions
    return


def commandline_err(msg):
    sys.stderr.write("%s: %s\n" % (PROGNAME, msg))
    sys.stderr.write("Try '%s --help' for more information.\n" % PROGNAME)
    return


class OptionClass:

    def __init__(self):
        self.datacol  = 1
        self.ymin     = 0.0
        self.ymax     = 1.0
        self.legend   = False
        self.linestyl = None
        self.logscale = None
        self.sep      = '\s+'
        self.cpu      = False
        self.thresh   = 0.0
        self.backend  = 'TkAgg'
        self.title    = None
        self.xlabel   = None
        self.ylabel   = None
        self.bw       = False


def parse_cmdline(arglist):
    """Parse argument list if given on the command line"""

    if len(arglist) == 0:
        usage()
        sys.exit(0)

    options = OptionClass()

    try: optlist, files = getopt.getopt(arglist, 'hl:c:x:y:t:',
                                        ["bw", "column=", "cpu", "help",
                                         "legend", "linestyle=",
                                         "sep=", "logscale=", "backend=",
                                         "thresh=", "title=",
                                         "xlabel=", "ylabel=",
                                        ])
    except getopt.error, e:
        commandline_err("%s" % str(e))
        sys.exit(1)

    if len(files) < 2:
        usage()
        sys.exit(0)

    opt_dict = {}

    for opt, arg in optlist:
        if   opt ==            "--bw" : opt_dict['bw'] = True
        elif opt in ("-c", "--column"): opt_dict['column'] = atoi(arg)
        elif opt ==           "--cpu" : opt_dict['cpu'] = True
        elif opt in ("-h",   "--help"):
            usage()
            sys.exit(0)
        elif opt ==     "--linestyle" : opt_dict['linestyle'] = arg
        elif opt ==        "--legend" : opt_dict['legend'] = True
        elif opt in ("-l",    "--log"): opt_dict['logscale'] = atoi(arg)
        elif opt ==           "--sep" : opt_dict['sep'] = arg
        elif opt ==       "--backend" : opt_dict['backend'] = arg
        elif opt ==        "--thresh" : opt_dict['threshold'] = atof(arg)
        elif opt in ("-t",  "--title"): opt_dict['title'] = arg
        elif opt in ("-x", "--xlabel"): opt_dict['xlabel'] = arg
        elif opt in ("-y", "--ylabel"): opt_dict['ylabel'] = arg

    return (opt_dict, files)


class MetricsClass:

    def __init__(self, solvers, opts):
        self.metric  = None
        self.nprobs  = []
        self.perf    = []
        self.solvers = solvers
        self.nsolvs  = len(solvers)
        self.opts    = opts

        map(self.add_solver, solvers)

        if opts.cpu and opts.thresh > 0.0:
            nmod = self.filter()
            print ' Updated %-d rows' % nmod

        print 'All solvers failed on %-d problems' % self.all_fail()

    def add_solver(self, fname):

        # Reg exp: Any line starting (ignoring white-space)
        # with a comment character. Also col sep.
        comment = re.compile(r'^[\s]*[%#]')
        column  = re.compile(self.opts.sep)

        # Grab the column from the file
        metrics = []
        file = open(fname, 'r')
        for line in file.readlines():
            if not comment.match(line):
                line = line.strip()
                cols = column.split(line)
                data = atof(cols[self.opts.datacol - 1])
                metrics.append(data)
        file.close()

        if self.metric is not None:
            self.metric = numpy.concatenate((self.metric, [metrics]))
        else:
            self.metric = numpy.array([metrics], dtype=numpy.float)

        # Current num of probs grabbed
        nprobs = len(metrics)
        if not self.nprobs: self.nprobs = nprobs
        elif self.nprobs != nprobs:
            commandline_error("All files must have same num of problems.")
            sys.exit(1)

    def filter(self):
        # Filter out problems on which all solvers succeeded but
        # produced a measure smaller than the threshold
        nmod = 0
        for prob in range(self.nprobs):
            all_small = masked_inside(self.metric[:,prob], 0, self.opts.thresh)
            if numpy.all(all_small.mask):
                self.metric[:,prob] = self.opts.thresh
                nmod += 1
        return nmod

    def all_fail(self):
        # Count the number of problems
        # on which all solvers failed
        nfail = 0
        for prob in range(self.nprobs):
            fail = ma.masked_less(self.metric[:,prob], 0.0)
            if numpy.all(fail.mask) == self.nsolvs:
                nfail += 1
        return nfail

    def prob_mets(self, prob):
        return ma.masked_less(self.metric[:,prob], 0.0)


class RatioClass:

    def __init__(self, MetricTable, opts):

        epsilon = 0.0
        if opts.cpu:
            epsilon = 0.01

        # Create empty ratio table
        nprobs = MetricTable.nprobs
        nsolvs = MetricTable.nsolvs
        self.ratios = ma.zeros((nprobs+1, nsolvs), dtype=numpy.float)

        # Compute best relative performance ratios across
        # solvers for each problem
        for prob in range(nprobs):
            metrics  = MetricTable.prob_mets(prob) + epsilon
            best_met = ma.minimum(metrics)
            self.ratios[prob+1,:] = metrics * (1.0 / best_met)

        # Sort each solvers performance ratios
        for solv in range(nsolvs):
            self.ratios[:,solv] = ma.sort(self.ratios[:,solv])

        # Compute largest ratio and use to replace failure entries
        self.maxrat = ma.maximum(self.ratios)
        self.ratios = ma.filled(self.ratios, numpy.inf) # 2 * self.maxrat)

    def solv_ratios(self, solver):
        return self.ratios[:,solver]


################
# Main program #
################

class PerformanceProfile:
    """
    A PerformanceProfile instance is a matplotlib representation of as many
    performance profiles as solvers given in argstring. The backend argument
    may be set to one of the acceptable matplotlib values. According to the
    value of backend, 'show' or 'savefig' should be called for visualization.

    Example 1:
        profile = PerformanceProfile(sys.argv[1:], backend = 'GTKAgg')
        profile.show()

        will display the profiles of solvers given on the command line
        in a matplotlib window using the GTKAgg backend.

    Example 2:
        profile = PerformanceProfile(sys.argv[1:], backend = 'PS')
        profile.savefig('my_profile.eps')

        will save the profiles in an encapsulated PostScript file.

    The optional values ymin and ymax may be altered to zoom in or out,
    when profiles are close to the axes boundary or when a region is of
    particular interest.

    Call usage() for the precise form of the argument argstring.
    """

    def __init__(self, solvers, **kwargs):

        self.solvers = solvers

        # Obtain options class with all default values
        self.opts = OptionClass()

        # Assign non-default options
        self.SetOptions(**kwargs)
        self.bw = self.opts.bw

        import matplotlib
        matplotlib.use(self.opts.backend)
        if matplotlib.__version__ < '0.65':
            import matplotlib.matlab as MM
        else:
            import matplotlib.pylab as MM

        self.linestyle = [ '-', '--', '-.', ':', '.', ',', 'o', '^', 'v', '<',
                           '>', 's', '+', 'x', 'D', 'd', '1', '2', '3', '4',
                           'h', 'H', 'p', '|', '_' ]
        self.nlstyles = len(self.linestyle)

        self.color = [ 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w' ]
        self.ncolors = len(self.color)

        self.metrics = MetricsClass(self.solvers, self.opts)
        self.pprofs  = RatioClass(self.metrics, self.opts)

        self.nprobs = self.metrics.nprobs
        self.nsolvs = self.metrics.nsolvs

        self.ax = MM.axes()

        # Generate the y-axis data
        self.ydata = numpy.arange(self.nprobs+1) * (1.0 / self.nprobs)

        # Set the x-axis ranges
        self.xmax = self.pprofs.maxrat + 1.0e-3
        self.epsilon = 0.0
        if self.opts.logscale is not None:
            self.mmplotcmd = self.ax.semilogx
            self.epsilon = 0.001
            self.xmin = 1.0
        else:
            self.mmplotcmd = self.ax.plot
            self.xmin = 0.0

        # Generate arguments for the gplot command
        self.profiles = []
        lscount = 0
        colcount = 0

        for s in range(self.nsolvs):
            if self.bw:
                curcolor = 'k'
            else:
                curcolor = self.color[(colcount % self.ncolors)]
            lstyle = self.linestyle[(lscount % self.nlstyles)]
            sname = self.solvers[s]
            srats = self.pprofs.solv_ratios(s)

            if self.opts.logscale is not None:
                #srats += self.epsilon
                srats = numpy.maximum(srats, self.epsilon)

            self.profiles.append(self.mmplotcmd(srats,
                                                self.ydata,
                                                curcolor + lstyle,
                                                linewidth=2,
                                                drawstyle='steps-pre'))
            if self.bw: lscount += 1
            if (lscount % self.nlstyles) == 0 and not self.bw:
                colcount += 1

        if self.opts.logscale is not None:
            self.ax.set_xscale('log', basex=self.opts.logscale)

        # Set legend if required
        if self.opts.legend:
            self.ax.legend(self.solvers, 'lower right')

        if self.opts.title:
            self.ax.set_title(self.opts.title)
        if self.opts.xlabel:
            self.ax.set_xlabel(self.opts.xlabel)
        if self.opts.ylabel:
            self.ax.set_ylabel(self.opts.ylabel)

        self.ax.set_xlim([self.xmin, self.xmax])
        self.ax.set_ylim([self.opts.ymin, self.opts.ymax])
        self.show = MM.show
        self.savefig = MM.savefig

    def SetOptions(self, **kwargs):
        keylist = kwargs.keys()
        if 'backend' in keylist:
            self.opts.backend = kwargs['backend']
        if 'ymin' in keylist:
            self.opts.ymin = kwargs['ymin']
        if 'ymax' in keylist:
            self.opts.ymax = kwargs['ymax']
        if 'column' in keylist:
            self.opts.datacol = kwargs['column']
        if 'cpu' in keylist:
            self.opts.cpu = kwargs['cpu']
        if 'legend' in keylist:
            self.opts.legend = kwargs['legend']
        if 'logscale' in keylist:
            self.opts.logscale = kwargs['logscale']
        if 'sep' in keylist:
            self.opts.sep = kwargs['sep']
        if 'threshold' in keylist:
            self.opts.thresh = kwargs['threshold']
        if 'title' in keylist:
            self.opts.title = kwargs['title']
        if 'xlabel' in keylist:
            self.opts.xlabel = kwargs['xlabel']
        if 'ylabel' in keylist:
            self.opts.ylabel = kwargs['ylabel']
        if 'bw' in keylist:
            self.opts.bw = kwargs['bw']
        return

###############################################################################

if __name__ == "__main__":

    # Usage from the command line
    (optlist, solvers) = parse_cmdline(sys.argv[1:])
    #print ' optlist = ', optlist
    #print ' solvers = ', solvers

    profile = PerformanceProfile(solvers, **optlist)

    # Usage from within a program
    #profile = PerformanceProfile(('solver1', 'solver2', 'solver3'),
    #                              backend = 'PS',
    #                              column = 2,
    #                              log = 2)
    #profile.savefig('mycoolprofile.eps')
    profile.show()

###############################################################################
