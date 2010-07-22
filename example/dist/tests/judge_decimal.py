#!/usr/bin/python

ABS_DELTA = 1e-5
REL_DELTA = 0

######################################################################

import sys
import re
import traceback
import optparse

DECIMAL_RE = re.compile(r'^-?(\d+\.?\d*|\.\d+)$', re.DOTALL)


def Judge(infile, difffile, outfile):
    try:
        difflines = open(difffile, 'rU').readlines()
        outlines = open(outfile, 'rU').readlines()
    except:
        print "Internal Error:"
        traceback.print_exc()
        return False
    n = len(difflines)
    accept = True
    for i in xrange(n):
        if not difflines[i].endswith("\n"):
            print "Internal Error: reference output file is not LF-terminated"
            accept = False
            return False
        difflines[i] = difflines[i][:-1]
        if not DECIMAL_RE.match(difflines[i]):
            print "Internal Error: reference output file format is incorrect at line %d" % (i+1)
            accept = False
    if not accept:
        return False
    for i in xrange(n):
        if i >= len(outlines):
            print "Line %d: missing output" % (i+1)
            print "  expected: \"%s\"" % diffline
            print "    output: <EOF>"
            accept = False
            continue
        diffline = difflines[i]
        outline = outlines[i].strip()
        if not DECIMAL_RE.match(outline):
            print "Line %d: output format is incorrect" % (i+1)
            print "  expected: \"%s\"" % diffline
            print "    output: \"%s\"" % outline
            accept = False
            continue
        x_diff = float(diffline)
        x_out = float(outline)
        x_delta = abs(x_diff - x_out)
        if x_delta > ABS_DELTA and x_delta > abs(x_diff) * REL_DELTA:
            print "Line %d: output range" % (i+1)
            print "  expected: \"%s\"" % diffline
            print "    output: \"%s\"" % outline
            print "     delta: %f" % x_delta
            accept = False
            continue
    for i in xrange(n, len(outlines)):
        outline = outlines[i].strip()
        print "Line %d: excessive output" % (i+1)
        print "  expected: <EOF>"
        print "    output: \"%s\"" % outline
        accept = False
    return accept


def ParseOptions():
    parser = optparse.OptionParser(usage="%prog [options] [infile] [difffile] [outfile] [resfile]")
    parser.add_option("-I", "--infile", dest="infile",
                      help="FILE is input file", metavar="FILE")
    parser.add_option("-D", "--difffile", dest="difffile",
                      help="FILE is reference output file", metavar="FILE")
    parser.add_option("-O", "--outfile", dest="outfile",
                      help="FILE is solution output file", metavar="FILE")
    parser.add_option("-o", "--resfile", dest="resfile",
                      help="print judge result to FILE", metavar="FILE")
    (options, args) = parser.parse_args()
    if len(args) >= 5:
        print "Internal Error: too many arguments given to judge script."
        sys.exit(1)
    if len(args) >= 4:
        options.resfile = args[3]
    if len(args) >= 3:
        options.outfile = args[2]
    if len(args) >= 2:
        options.difffile = args[1]
    if len(args) >= 1:
        options.infile = args[0]
    if options.infile is None:
        print "Internal Error: input file is not specified"
        sys.exit(1)
    if options.difffile is None:
        print "Internal Error: reference output file is not specified"
        sys.exit(1)
    if options.outfile is None:
        print "Internal Error: solution output file is not specified"
        sys.exit(1)
    return options


def main():
    options = ParseOptions()
    if options.resfile is not None:
        sys.stdout = sys.stderr = open(options.resfile, 'w')
    res = Judge(options.infile, options.difffile, options.outfile)
    if not res:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()

