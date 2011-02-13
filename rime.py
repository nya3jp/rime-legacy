#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2010 Rime Authors.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Author: Shuhei Takahashi
#

from __future__ import with_statement

import datetime
import optparse
import os
import pickle
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback


HELP_MESSAGE = """\
Usage: rime.py COMMAND [OPTIONS] [DIR]

Commands:
  build     build solutions/tests
  test      run tests
  clean     clean up built files
  help      show this help message and exit

Options:
  -h, --help         show this help message and exit
  -C, --cache-tests  cache test results
  -d, --debug        print debug messages
"""


class RimeConfigurationError(Exception):
  pass


class FileNames(object):
  """
  Set of filename constants.
  """

  RIMEROOT_FILE = 'RIMEROOT'
  PROBLEM_FILE = 'PROBLEM'
  SOLUTION_FILE = 'SOLUTION'
  TESTS_FILE = 'TESTS'

  STAMP_FILE = '.stamp'

  IN_EXT = '.in'
  DIFF_EXT = '.diff'
  OUT_EXT = '.out'
  EXE_EXT = '.exe'
  JUDGE_EXT = '.judge'
  CACHE_EXT = '.cache'
  LOG_EXT = '.log'
  VALIDATION_EXT = '.validation'

  RIME_OUT_DIR = 'rime-out'
  TESTS_DIR = 'tests'
  TESTS_PACKED_DIR = 'tests.packed'
  TESTS_PACKED_TARBALL = 'tests.tar.gz'
  CONCAT_PREFIX = '.ALL'
  CONCAT_INFILE = CONCAT_PREFIX + IN_EXT
  CONCAT_DIFFFILE = CONCAT_PREFIX + DIFF_EXT
  SEPARATOR_FILE = 'seperator'
  TERMINATOR_FILE = 'terminator'


class FileUtil(object):
  """
  Set of utility functions to manipulate files.
  """

  @classmethod
  def CopyFile(cls, src, dst):
    shutil.copy(src, dst)

  @classmethod
  def MakeDir(cls, dir):
    if not os.path.isdir(dir):
      os.makedirs(dir)

  @classmethod
  def CopyTree(cls, src, dst):
    cls.MakeDir(dst)
    files = cls.ListDir(src, True)
    for f in files:
      srcpath = os.path.join(src, f)
      dstpath = os.path.join(dst, f)
      if os.path.isdir(srcpath):
        cls.MakeDir(dstpath)
      else:
        cls.CopyFile(srcpath, dstpath)

  @classmethod
  def RemoveTree(cls, dir):
    if os.path.exists(dir):
      shutil.rmtree(dir)

  @classmethod
  def GetModified(cls, file):
    try:
      return datetime.datetime.fromtimestamp(os.path.getmtime(file))
    except:
      return datetime.datetime.min

  @classmethod
  def GetLastModifiedUnder(cls, dir):
    return max([cls.GetModified(os.path.join(dir, name))
               for name in (cls.ListDir(dir, True) + [dir])])

  @classmethod
  def CreateEmptyFile(cls, file):
    open(file, 'w').close()

  @classmethod
  def ListDir(cls, dir, recursive=False):
    files = []
    try:
      files = filter(lambda x: not x.startswith("."),
                     os.listdir(dir))
      if recursive:
        for subfile in files[:]:
          subdir = os.path.join(dir, subfile)
          if os.path.isdir(subdir):
            files += [os.path.join(subfile, s)
                      for s in cls.ListDir(subdir, True)]
    except:
      pass
    return files

  @classmethod
  def PickleSave(cls, obj, file):
    with open(file, 'w') as f:
      pickle.dump(obj, f)

  @classmethod
  def PickleLoad(cls, file):
    with open(file, 'r') as f:
      obj = pickle.load(f)
      return obj

  @classmethod
  def ConvPath(cls, path):
    if not os.uname()[0].lower().startswith('cygwin'):
      return path
    try:
      p = subprocess.Popen(['cygpath', '-wp', path], stdout=subprocess.PIPE)
      newpath = p.communicate()[0].rstrip('\r\n')
      if p.returncode == 0:
        return newpath
    except:
      pass
    return path

  @classmethod
  def LocateBinary(cls, name):
    if 'PATH' in os.environ:
      paths = os.environ['PATH']
    else:
      paths = os.defpath
    for path in paths.split(os.pathsep):
      bin = os.path.join(path, name)
      if os.path.isfile(bin) and os.access(bin, os.X_OK):
        return bin
    return None

  @classmethod
  def OpenNull(cls):
    if not hasattr(cls, '_devnull'):
      cls._devnull = open(os.devnull, 'w')
    return cls._devnull

  @classmethod
  def ReadFile(cls, name):
    try:
      with open(name, 'r') as f:
        return f.read()
    except:
      return None

  @classmethod
  def WriteFile(cls, content, name):
    try:
      with open(name, 'w') as f:
        f.write(content)
      return True
    except:
      return False

  @classmethod
  def AppendFile(cls, content, name):
    try:
      with open(name, 'a') as f:
        f.write(content)
        return True
    except:
      return False


class Console(object):
  """
  Provides common interface for printing to console.
  """

  # Capabilities of the console.
  class Capability(object):
    overwrite = False
    color = False
  cap = Capability()

  # Console codes.
  # Set in _SetupColors() if available.
  BOLD = ''
  RED = ''
  GREEN = ''
  YELLOW = ''
  BLUE = ''
  MAGENTA = ''
  CYAN = ''
  WHITE = ''
  NORMAL = ''
  UP = ''
  KILL = ''

  # State for overwriting.
  last_progress = False

  @classmethod
  def Init(cls):
    """
    Check availability of console codes.
    """
    if not sys.stdout.isatty():
      return
    try:
      import curses
      curses.setupterm()
      if cls._tigetstr('cuu1'):
        cls.cap.overwrite = True
      if cls._tigetstr('setaf'):
        cls.cap.color = True
        cls._SetupColors()
    except:
      pass

  @classmethod
  def _tigetstr(cls, name):
    import curses
    return curses.tigetstr(name) or ''

  @classmethod
  def _SetupColors(cls):
    cls.BOLD = '\x1b[1m'
    cls.RED = '\x1b[31m'
    cls.GREEN = '\x1b[32m'
    cls.YELLOW = '\x1b[33m'
    cls.BLUE = '\x1b[34m'
    cls.MAGENTA = '\x1b[35m'
    cls.CYAN = '\x1b[36m'
    cls.WHITE = '\x1b[37m'
    cls.NORMAL = '\x1b[0m'
    cls.UP = '\x1b[1A'
    cls.KILL = '\x1b[K'

  @classmethod
  def Print(cls, *args, **kwargs):
    """
    Print one line.
    Each argument is either ordinal string or control code.
    """
    progress = bool(kwargs.get('progress'))
    msg = "".join(args)
    if cls.last_progress and cls.cap.overwrite:
      print cls.UP + "\r" + msg + cls.KILL
    else:
      print msg
    cls.last_progress = progress

  @classmethod
  def PrintAction(cls, action, obj, *args, **kwargs):
    """
    Utility function to print actions.
    """
    real_args = [cls.GREEN, "[" + action.center(10) + "]", cls.NORMAL]
    if obj:
      real_args += [" ", obj.fullname]
    if args:
      if obj:
        real_args += [":"]
      real_args += [" "] + list(args)
    cls.Print(*real_args, **kwargs)

  @classmethod
  def PrintError(cls, msg):
    """
    Utility function to print errors.
    """
    cls.Print(cls.RED, "ERROR:", cls.NORMAL, " ", msg)

  @classmethod
  def PrintWarning(cls, msg):
    """
    Utility function to print warnings.
    """
    cls.Print(cls.YELLOW, "WARNING:", cls.NORMAL, " ", msg)

  @classmethod
  def PrintLog(cls, log):
    """
    Barely print messages.
    Used to print logs such as compiler's output.
    """
    if log is None:
      return
    for line in log.splitlines():
      cls.Print("> ", line)

# Call Init() on load time.
Console.Init()



class ErrorRecorder(object):
  """
  Accumurates errors/warnings and print summary of them.
  """

  def __init__(self, ctx):
    self.ctx = ctx
    self.errors = []
    self.warnings = []

  def Error(self, source, reason, quiet=False, stack_offset=0):
    """
    Emit an error.
    If quiet is True it is not printed immediately, but shown in summary.
    """
    if source:
      msg = "%s: %s" % (source.fullname, reason)
    else:
      msg = reason
    if self.ctx.options.debug >= 1:
      msg += " [" + self._FormatStack(stack_offset) + "]"
    self.errors.append(msg)
    if not quiet:
      Console.PrintError(msg)

  def Warning(self, source, reason, quiet=False, stack_offset=0):
    """
    Emit an warning.
    If quiet is True it is not printed immediately, but shown in summary.
    """
    if source:
      msg = "%s: %s" % (source.fullname, reason)
    else:
      msg = reason
    if self.ctx.options.debug >= 1:
      msg += " [" + self._FormatStack(stack_offset) + "]"
    self.warnings.append(msg)
    if not quiet:
      Console.PrintWarning(msg)

  def Exception(self, source, e=None, quiet=False, stack_offset=0):
    """
    Emit an exception without aborting.
    If e is not given, use current context.
    If quiet is True it is not printed immediately, but shown in summary.
    """
    if e is None:
      e = sys.exc_info()[1]
    self.Error(source, str(e), quiet=quiet, stack_offset=stack_offset+1)

  def _FormatStack(self, stack_offset):
    stack = traceback.extract_stack()
    (filename, lineno, modulename, code) = stack[-3-stack_offset]
    return 'File "%s", line %d, in %s' % (filename, lineno, modulename)

  def HasError(self):
    return bool(self.errors)

  def HasWarning(self):
    return bool(self.warnings)

  def PrintSummary(self):
    for e in self.errors:
      Console.PrintError(e)
    for e in self.warnings:
      Console.PrintWarning(e)
    Console.Print(("Total %d errors, %d warnings" %
                   (len(self.errors), len(self.warnings))))



class RimeContext(object):
  """
  Context object of rime.
  """

  def __init__(self, options):
    self.options = options
    self.errors = ErrorRecorder(self)



class SingleCaseResult(object):
  """
  Test result of a single solution versus a single test case.
  """

  def __init__(self, verdict, time):
    # Pre-defined list of verdicts are listed in TestResult.
    self.verdict = verdict
    self.time = time



class TestResult(object):
  """
  Test result of a single solution.
  This includes sub-results for each test case.
  """

  # Note: verdict can be set different from any of these;
  # in that case, you should treat it as System Error.
  NA = "-"
  AC = "Accepted"
  WA = "Wrong Answer"
  TLE = "Time Limit Exceeded"
  RE = "Runtime Error"
  ERR = "System Error"

  def __init__(self, problem, solution, files):
    """
    Construct with empty results.
    """
    self.problem = problem
    self.solution = solution
    self.files = files[:]
    self.cases = dict(
      [(f, SingleCaseResult(TestResult.NA, None))
       for f in files])
    self.good = None
    self.passed = None
    self.detail = None
    self.ruling_file = None
    self.cached = False

  def IsTimeStatsAvailable(self):
    """
    Checks if time statistics are available.
    """
    return (self.files and all([c.verdict == TestResult.AC for c in self.cases.values()]))

  def GetTimeStats(self):
    """
    Get time statistics.
    """
    if (FileNames.CONCAT_INFILE in self.cases and
        self.cases[FileNames.CONCAT_INFILE].time is not None):
      return "(%.2f/%.2f/%.2f)" % (self.GetMaxTime(), self.GetTotalTime(),
                                   self.cases[FileNames.CONCAT_INFILE].time)
    return "(%.2f/%.2f)" % (self.GetMaxTime(), self.GetTotalTime())

  def GetMaxTime(self):
    """
    Get maximum time.
    All case should be accepted.
    """
    return max([c.time for k, c in self.cases.items() if not k.startswith(".")])

  def GetTotalTime(self):
    """
    Get total time.
    All case should be accepted.
    """
    return sum([c.time for k, c in self.cases.items() if not k.startswith(".")])

  @classmethod
  def CompareForListing(cls, a, b):
    """
    Compare two TestResult for display-ordering.
    """
    if a.problem.name != b.problem.name:
      return cmp(a.problem.name, b.problem.name)
    reference_solution = a.problem.reference_solution
    if a.solution is reference_solution:
      return -1
    if b.solution is reference_solution:
      return +1
    if a.solution.IsCorrect() != b.solution.IsCorrect():
      return -cmp(a.solution.IsCorrect(), b.solution.IsCorrect())
    return cmp(a.solution.name, b.solution.name)



class RunResult(object):
  """
  Result of a single run.
  Note that this is not judgement result but just execution status.
  """

  OK = "OK"
  NG = "Exited Abnormally"
  RE = "Runtime Error"
  TLE = "Time Limit Exceeded"
  PM = "Prerequisite Missing"

  def __init__(self, status, time):
    self.status = status
    self.time = time



class Code(object):
  """
  Source code.
  Supports operations such as compile, run, clean.
  """

  # Set to True if deriving class of source code does not
  # require compilation (e.g. script language).
  QUIET_COMPILE = False

  def __init__(self):
    pass

  def Compile(self):
    raise NotImplementedError()

  def Run(self, args, cwd, input, output, timeout, redirect_error=False):
    raise NotImplementedError()

  def Clean(self):
    raise NotImplementedError()



class FileBasedCode(Code):
  """
  Source code which is based on files.
  """

  # Should be set in each instance.
  src_name = None
  log_name = None
  src_dir = None
  out_dir = None
  prereqs = None

  def __init__(self, src_name, src_dir, out_dir, prereqs):
    super(FileBasedCode, self).__init__()
    self.src_name = src_name
    self.src_dir = src_dir
    self.out_dir = out_dir
    self.prereqs = prereqs
    self.log_name = os.path.splitext(src_name)[0] + FileNames.LOG_EXT

  def MakeOutDir(self):
    """
    Create output directory.
    """
    FileUtil.MakeDir(self.out_dir)

  def Compile(self):
    """
    Compile the code and return (RunResult, log) pair.
    """
    try:
      for name in self.prereqs:
        if not FileUtil.LocateBinary(name):
          return RunResult("%s: %s" % (RunResult.PM, name), None)
      self.MakeOutDir()
      return self._ExecForCompile(args=self.compile_args)
    except Exception, e:
      return RunResult(str(e), None)

  def Run(self, args, cwd, input, output, timeout, redirect_error=False):
    """
    Run the code and return RunResult.
    """
    try:
      return self._ExecForRun(
        args=self.run_args+args, cwd=cwd,
        input=input, output=output, timeout=timeout,
        redirect_error=redirect_error)
    except Exception, e:
      return RunResult(str(e), None)

  def Clean(self):
    """
    Clean the output directory.
    Return an exception object on error.
    """
    try:
      FileUtil.RemoveTree(self.out_dir)
      return None
    except Exception, e:
      return e

  def _ExecForCompile(self, args):
    with open(os.path.join(self.out_dir, self.log_name), 'w') as outfile:
      return self._ExecInternal(
        args=args, cwd=self.src_dir,
        stdin=FileUtil.OpenNull(), stdout=outfile, stderr=subprocess.STDOUT)

  def ReadCompileLog(self):
    return FileUtil.ReadFile(os.path.join(self.out_dir, self.log_name))

  def _ExecForRun(self, args, cwd, input, output, timeout, redirect_error):
    with open(input, 'r') as infile:
      with open(output, 'w') as outfile:
        if redirect_error:
          errfile = subprocess.STDOUT
        else:
          errfile = FileUtil.OpenNull()
        return self._ExecInternal(
          args=args, cwd=cwd,
          stdin=infile, stdout=outfile, stderr=errfile, timeout=timeout)

  def _ExecInternal(self, args, cwd, stdin, stdout, stderr, timeout=None):
    start_time = time.time()
    p = subprocess.Popen(args, cwd=cwd,
                         stdin=stdin, stdout=stdout, stderr=stderr)
    if timeout is not None:
      timer = threading.Timer(timeout,
                              lambda: os.kill(p.pid, signal.SIGKILL))
      timer.start()
    code = p.wait()
    end_time = time.time()
    if timeout is not None:
      timer.cancel()
    if code == 0:
      status = RunResult.OK
    elif code == -(signal.SIGKILL):
      status = RunResult.TLE
    elif code < 0:
      status = RunResult.RE
    else:
      status = RunResult.NG
    return RunResult(status, end_time-start_time)



class CCode(FileBasedCode):

  def __init__(self, src_name, src_dir, out_dir, flags):
    super(CCode, self).__init__(
      src_name=src_name, src_dir=src_dir, out_dir=out_dir,
      prereqs=['gcc'])
    self.exe_name = os.path.splitext(src_name)[0] + FileNames.EXE_EXT
    self.compile_args = ['gcc',
                         '-o', os.path.join(out_dir, self.exe_name),
                         src_name] + flags
    self.run_args = [os.path.join(out_dir, self.exe_name)]



class CXXCode(FileBasedCode):

  def __init__(self, src_name, src_dir, out_dir, flags):
    super(CXXCode, self).__init__(
      src_name=src_name, src_dir=src_dir, out_dir=out_dir,
      prereqs=['g++'])
    self.exe_name = os.path.splitext(src_name)[0] + FileNames.EXE_EXT
    self.compile_args = ['g++',
                         '-o', os.path.join(out_dir, self.exe_name),
                         src_name] + flags
    self.run_args = [os.path.join(out_dir, self.exe_name)]



class JavaCode(FileBasedCode):

  def __init__(self, src_name, src_dir, out_dir,
               encoding, mainclass, compile_flags, run_flags):
    super(JavaCode, self).__init__(
      src_name=src_name, src_dir=src_dir, out_dir=out_dir,
      prereqs=['javac', 'java'])
    self.encoding = encoding
    self.mainclass = mainclass
    self.compile_args = (['javac', '-encoding', encoding,
                          '-d', FileUtil.ConvPath(out_dir)] +
                         compile_flags + [src_name])
    self.run_args = (['java', '-Dline.separator=\n',
                      '-cp', FileUtil.ConvPath(out_dir)] +
                     run_flags + [mainclass])



class ScriptCode(FileBasedCode):

  QUIET_COMPILE = True

  def __init__(self, src_name, src_dir, out_dir, interpreter):
    super(ScriptCode, self).__init__(
      src_name=src_name, src_dir=src_dir, out_dir=out_dir,
      prereqs=[interpreter])
    self.interpreter = interpreter
    self.run_args = [interpreter, os.path.join(out_dir, src_name)]

  def Compile(self):
    try:
      self.MakeOutDir()
      FileUtil.CopyFile(os.path.join(self.src_dir, self.src_name),
                        os.path.join(self.out_dir, self.src_name))
      return RunResult(RunResult.OK, 0.0)
    except Exception, e:
      FileUtil.WriteFile(str(e), os.path.join(self.out_dir, self.log_name))
      return RunResult(RunResult.RE, None)



class DiffCode(Code):

  QUIET_COMPILE = True

  def __init__(self):
    super(DiffCode, self).__init__()
    self.log_name = 'diff.log'

  def Compile(self):
    if not FileUtil.LocateBinary('diff'):
      return RunResult("%s: diff" % RunResult.PM, None)
    return RunResult(RunResult.OK, 0.0)

  def Run(self, args, cwd, input, output, timeout, redirect_error=False):
    parser = optparse.OptionParser()
    parser.add_option('-i', '--infile', dest='infile')
    parser.add_option('-d', '--difffile', dest='difffile')
    parser.add_option('-o', '--outfile', dest='outfile')
    (options, pos_args) = parser.parse_args([''] + args)
    run_args = ['diff', '-u', options.difffile, options.outfile]
    with open(input, 'r') as infile:
      with open(output, 'w') as outfile:
        if redirect_error:
          errfile = subprocess.STDOUT
        else:
          errfile = FileUtil.OpenNull()
        try:
          ret = subprocess.call(run_args, cwd=cwd,
                                stdin=infile, stdout=outfile, stderr=errfile)
        except OSError:
          return RunResult(RunResult.RE, None)
        if ret == 0:
          return RunResult(RunResult.OK, 0.0)
        if ret > 0:
          return RunResult(RunResult.NG, None)
        return RunResult(RunResult.RE, None)

  def Clean(self):
    return True


class ConfigurableObject(object):
  """
  Base class for configurable, i.e., associated with directory-config
  pair under Rime's management, classes.
  """

  # Class-specific config file name.
  # Should be set in derived classes.
  CONFIG_FILE = None

  @classmethod
  def CanLoadFrom(cls, base_dir):
    """
    Checks if this kind of object is constructable from the specified dir.
    """
    return os.path.isfile(os.path.join(base_dir, cls.CONFIG_FILE))

  def __init__(self, name, base_dir, parent):
    """
    Constructs a new unconfigured object.
    """
    self.name = name
    self.base_dir = base_dir
    self.parent = parent
    if parent is None:
      self.root = self
    else:
      self.root = parent.root
    # Set full name.
    # Full name is normally path-like string separated with "/".
    if name is None:
      self.fullname = None
    elif parent is None or parent.fullname is None:
      self.fullname = name
    else:
      self.fullname = parent.fullname + "/" + name
    # Locate config file.
    self.config_file = os.path.join(base_dir, self.CONFIG_FILE)
    self.real_config_file = self.config_file
    if not os.path.isfile(self.real_config_file):
      self.real_config_file = os.devnull
    self._loaded = False

  def Load(self, ctx):
    """
    Loads configurations and do setups.
    """
    assert not self._loaded
    self._loaded = True
    # Setup input/output directionaries and evaluate config.
    self.config = dict()
    self._export_dict = dict()
    self._dependencies = []
    self._PreLoad(ctx)
    # Export functions marked with @Export.
    for name in dir(self):
      try:
        attr = getattr(self, name)
        self._export_dict[attr.im_func._export_config] = attr
      except:
        pass
    # Evaluate config.
    script = FileUtil.ReadFile(self.real_config_file)
    code = compile(script, self.config_file, 'exec')
    # TODO: exception handling here
    exec(code, self._export_dict, self.config)
    self._PostLoad(ctx)

  def _PreLoad(self, ctx):
    """
    Called just before evaluation of config.
    Should setup symbols to export via self._export_dict.
    """
    pass

  def _PostLoad(self, ctx):
    """
    Called just after evaluation of config.
    Do some post-processing of configs here.
    """
    pass

  def FindByBaseDir(self, base_dir):
    """
    Search whole subtree under this object and
    return the object with matching base_dir.
    Subclasses may want to override this for recursive search.
    """
    if self.base_dir == base_dir:
      return self
    return None

  def GetLastModified(self):
    """
    Get timestamp of this target.
    """
    stamp = FileUtil.GetLastModifiedUnder(self.src_dir)
    for d in self._dependencies:
      stamp = max(stamp, d.GetLastModified())
    return stamp

  def _AddDependency(self, obj):
    """
    Add a dependency of this target to another target.
    This is used for deciding last modified stamp.
    """
    assert issubclass(obj.__class__, ConfigurableObject)
    self._dependencies.append(obj)

  @classmethod
  def Export(cls, name):
    """
    Decorator to mark methods "to be exported".
    """
    def ExportImpl(f):
      f._export_config = name
    return ExportImpl


class BuildableObject(ConfigurableObject):
  """
  ConfigurableObject with its dedicated output directory.
  """

  def __init__(self, name, base_dir, out_dir, parent):
    super(BuildableObject, self).__init__(name, base_dir, parent)
    self.src_dir = base_dir
    self.out_dir = out_dir
    self.stamp_file = os.path.join(self.out_dir, FileNames.STAMP_FILE)

  def SetCacheStamp(self, ctx):
    """
    Update stamp file.
    """
    try:
      FileUtil.CreateEmptyFile(self.stamp_file)
      return True
    except:
      ctx.errors.Exception(self)
      return False

  def GetCacheStamp(self):
    """
    Get timestamp of stamp file.
    Returns minimum datetime if not available.
    """
    return FileUtil.GetModified(self.stamp_file)

  def IsBuildCached(self):
    """
    Check if cached build is not staled.
    """
    stamp_mtime = self.GetCacheStamp()
    src_mtime = self.GetLastModified()
    return (src_mtime < stamp_mtime)

  def _AddCodeRegisterer(self, field_name, command_name):
    """
    Export {c,cxx,java,script}_hogehoge functions.
    """
    multiple = (type(getattr(self, field_name)) is list)
    def GenericRegister(code):
      field = getattr(self, field_name)
      if not multiple:
        if field is not None:
          # TODO: remove this all_result reference.
          raise RimeConfigurationError(
            "Multiple %ss specified" % command_name)
        setattr(self, field_name, code)
      if multiple:
        field.append(code)
    def CRegister(src, flags=['-Wall', '-g', '-O2', '-lm']):
      GenericRegister(CCode(
        src_name=src,
        src_dir=self.src_dir, out_dir=self.out_dir,
        flags=flags))
    def CXXRegister(src, flags=['-Wall', '-g', '-O2']):
      GenericRegister(CXXCode(
        src_name=src,
        src_dir=self.src_dir, out_dir=self.out_dir,
        flags=flags))
    def JavaRegister(
      src, encoding='UTF-8', mainclass='Main',
      compile_flags=[], run_flags=['-Xmx256M']):
      GenericRegister(JavaCode(
        src_name=src,
        src_dir=self.src_dir, out_dir=self.out_dir,
        encoding=encoding, mainclass=mainclass,
        compile_flags=compile_flags,
        run_flags=run_flags))
    def ScriptRegister(src, interpreter='perl'):
      GenericRegister(ScriptCode(
        src_name=src,
        src_dir=self.src_dir, out_dir=self.out_dir,
        interpreter=interpreter))
    registers = [("c_" + command_name, CRegister),
                 ("cxx_" + command_name, CXXRegister),
                 ("java_" + command_name, JavaRegister),
                 ("script_" + command_name, ScriptRegister)]
    for (name, func) in registers:
      self._export_dict[name] = func
      setattr(self, name, func)



class RimeRoot(ConfigurableObject):
  """
  Represent the root of Rime tree.
  """

  CONFIG_FILE = FileNames.RIMEROOT_FILE

  def _PreLoad(self, ctx):
    self._export_dict["root"] = self

  def _PostLoad(self, ctx):
    self.concat_test = self.config.get('CONCAT_TEST')
    # Chain-load problems.
    self.problems = []
    for name in FileUtil.ListDir(self.base_dir):
      path = os.path.join(self.base_dir, name)
      if Problem.CanLoadFrom(path):
        problem = Problem(name, path, self)
        problem.Load(ctx)
        self.problems.append(problem)
    self.problems.sort(lambda a, b: cmp((a.id, a.name), (b.id, b.name)))

  def FindByBaseDir(self, base_dir):
    if self.base_dir == base_dir:
      return self
    for problem in self.problems:
      obj = problem.FindByBaseDir(base_dir)
      if obj:
        return obj
    return None

  def Build(self, ctx):
    """
    Build all.
    """
    success = True
    for problem in self.problems:
      if not problem.Build(ctx):
        success = False
    return success

  def Test(self, ctx):
    """
    Test all.
    """
    results = []
    for problem in self.problems:
      results.extend(problem.Test(ctx))
    return results

  def Pack(self, ctx):
    """
    Pack all.
    """
    success = True
    for problem in self.problems:
      if not problem.Pack(ctx):
        success = False
    return success

  def Clean(self, ctx):
    """
    Clean all.
    """
    success = True
    for problem in self.problems:
      if not problem.Clean(ctx):
        success = False
    return success



class Problem(BuildableObject):
  """
  Represent a single problem.
  """

  CONFIG_FILE = FileNames.PROBLEM_FILE

  def __init__(self, name, base_dir, parent):
    super(Problem, self).__init__(
      name, base_dir, os.path.join(base_dir, FileNames.RIME_OUT_DIR), parent)

  def _PreLoad(self, ctx):
    self._export_dict["problem"] = self
    self._export_dict["root"] = self.root

  def _PostLoad(self, ctx):
    # Read time limit.
    if 'TIME_LIMIT' not in self.config:
      ctx.errors.Error(self, "Time limit is not specified")
    else:
      self.timeout = self.config['TIME_LIMIT']
    # Decide ID.
    self.id = self.config.get('ID', self.name)
    # Chain-load solutions.
    self.solutions = []
    for name in sorted(FileUtil.ListDir(self.base_dir)):
      path = os.path.join(self.base_dir, name)
      if Solution.CanLoadFrom(path):
        solution = Solution(name, path, self)
        solution.Load(ctx)
        self.solutions.append(solution)
    self._SelectReferenceSolution(ctx)
    # Chain-load tests.
    self.tests = Tests(
      FileNames.TESTS_DIR,
      os.path.join(self.base_dir, FileNames.TESTS_DIR),
      self)
    self.tests.Load(ctx)

  def _SelectReferenceSolution(self, ctx):
    """
    Select a reference solution.
    """
    self.reference_solution = None
    if 'REFERENCE_SOLUTION' not in self.config:
      # If not explicitly specified, select one which is
      # not marked as incorrect.
      for solution in self.solutions:
        if solution.IsCorrect():
          self.reference_solution = solution
          break
    else:
      # If explicitly specified, just use it.
      reference_solution_name = self.config['REFERENCE_SOLUTION']
      for solution in self.solutions:
        if solution.name == reference_solution_name:
          self.reference_solution = solution
          break
      if self.reference_solution is None:
        ctx.errors.Error(
          self,
          ("Reference solution \"%s\" does not exist" %
           reference_solution_name))

  def FindByBaseDir(self, base_dir):
    if self.base_dir == base_dir:
      return self
    for solution in self.solutions:
      obj = solution.FindByBaseDir(base_dir)
      if obj:
        return obj
    return self.tests.FindByBaseDir(base_dir)

  def Build(self, ctx):
    """
    Build all solutions and tests.
    """
    success = True
    for solution in self.solutions:
      if not solution.Build(ctx):
        success = False
    if not self.tests.Build(ctx):
      success = False
    return success

  def Test(self, ctx):
    """
    Run tests.
    """
    return self.tests.Test(ctx)

  def Pack(self, ctx):
    """
    Pack tests.
    """
    return self.tests.Pack(ctx)

  def Clean(self, ctx):
    """
    Clean all solutions and tests.
    """
    Console.PrintAction("CLEAN", self)
    success = True
    if success:
      try:
        FileUtil.RemoveTree(self.out_dir)
      except:
        ctx.errors.Exception(self)
        success = False
    return success



class Tests(BuildableObject):
  """
  Represent a test set for a problem.
  """

  CONFIG_FILE = FileNames.TESTS_FILE

  def __init__(self, name, base_dir, parent):
    super(Tests, self).__init__(
      name, base_dir, os.path.join(parent.out_dir, FileNames.TESTS_DIR), parent)
    self.problem = parent
    self.pack_dir = os.path.join(self.problem.out_dir, FileNames.TESTS_PACKED_DIR)
    self.separator_file = os.path.join(self.out_dir, FileNames.SEPARATOR_FILE)
    self.terminator_file = os.path.join(self.out_dir, FileNames.TERMINATOR_FILE)

  def _PreLoad(self, ctx):
    self.concat_test = self.root.concat_test
    self.generators = []
    self.validators = []
    self.judge = None
    self._AddCodeRegisterer('generators', 'generator')
    self._AddCodeRegisterer('validators', 'validator')
    self._AddCodeRegisterer('judge', 'judge')
    if not os.path.isfile(self.config_file):
      ctx.errors.Warning(self,
                         "%s does not exist" % self.CONFIG_FILE)
    self._export_dict["tests"] = self
    self._export_dict["root"] = self.root
    self._export_dict["problem"] = self.problem

  def _PostLoad(self, ctx):
    # TODO: print warnings if no validator / judge is specified.
    if self.judge is None:
      self.judge = DiffCode()
    if self.problem.reference_solution:
      self._AddDependency(self.problem.reference_solution)

  def Build(self, ctx):
    """
    Build tests.
    """
    if self.IsBuildCached():
      return True
    if not self._InitOutputDir(ctx):
      return False
    if not self._CompileGenerator(ctx):
      return False
    if not self._CompileValidator(ctx):
      return False
    if not self._CompileJudge(ctx):
      return False
    if not self._RunGenerator(ctx):
      return False
    if not self._RunValidator(ctx):
      return False
    if self.ListInputFiles():
      if not self._CompileReferenceSolution(ctx):
        return False
      if not self._RunReferenceSolution(ctx):
        return False
      if not self._GenerateConcatTest(ctx):
        return False
    if not self.SetCacheStamp(ctx):
      return False
    return True

  def _InitOutputDir(self, ctx):
    """
    Initialize output directory.
    """
    try:
      FileUtil.RemoveTree(self.out_dir)
      FileUtil.CopyTree(self.src_dir, self.out_dir)
      return True
    except:
      ctx.errors.Exception(self)
      return False

  def _CompileGenerator(self, ctx):
    """
    Compile all input generators.
    """
    for generator in self.generators:
      if not generator.QUIET_COMPILE:
        Console.PrintAction("COMPILE", self, generator.src_name)
      res = generator.Compile()
      if res.status != RunResult.OK:
        ctx.errors.Error(self,
                         "%s: Compile Error (%s)" % (generator.src_name, res.status))
        Console.PrintLog(generator.ReadCompileLog())
        return False
    return True

  def _RunGenerator(self, ctx):
    """
    Run all input generators.
    """
    for generator in self.generators:
      Console.PrintAction("GENERATE", self, generator.src_name)
      res = generator.Run(
        args=[], cwd=self.out_dir,
        input=os.devnull, output=os.devnull, timeout=None)
      if res.status != RunResult.OK:
        ctx.errors.Error(self,
                         "%s: %s" % (generator.src_name, res.status))
        return False
    return True

  def _CompileValidator(self, ctx):
    """
    Compile input validators.
    """
    for validator in self.validators:
      if not validator.QUIET_COMPILE:
        Console.PrintAction("COMPILE", self, validator.src_name)
      res = validator.Compile()
      if res.status != RunResult.OK:
        ctx.errors.Error(self,
                         "%s: Compile Error (%s)" % (validator.src_name, res.status))
        Console.PrintLog(validator.ReadCompileLog())
        return False
    return True

  def _RunValidator(self, ctx):
    """
    Run input validators.
    """
    if not self.validators:
      Console.PrintAction("VALIDATE", self, "skipping: validator unavailable")
      ctx.errors.Warning(self, "Validator Unavailable")
      return True
    for validator in self.validators:
      Console.PrintAction("VALIDATE", self, progress=True)
      infiles = self.ListInputFiles()
      for (i, infile) in enumerate(infiles):
        Console.PrintAction(
          "VALIDATE", self,
          "[%d/%d] %s" % (i+1, len(infiles), infile),
          progress=True)
        validationfile = os.path.splitext(infile)[0] + FileNames.VALIDATION_EXT
        res = validator.Run(
          args=[], cwd=self.out_dir,
          input=os.path.join(self.out_dir, infile),
          output=os.path.join(self.out_dir, validationfile),
          timeout=None,
          redirect_error=True)
        if res.status == RunResult.NG:
          ctx.errors.Error(self,
                           "%s: Validation Failed" % infile)
          log = FileUtil.ReadFile(os.path.join(self.out_dir, validationfile))
          Console.PrintLog(log)
          return False
        elif res.status != RunResult.OK:
          ctx.errors.Error(self,
                           "%s: Validator Failed: %s" % (infile, res.status))
          return False
      Console.PrintAction("VALIDATE", self, "OK")
    return True

  def _CompileJudge(self, ctx):
    """
    Compile judge.
    """
    if self.judge is None:
      return True
    if not self.judge.QUIET_COMPILE:
      Console.PrintAction("COMPILE",
                          self,
                          self.judge.src_name)
    res = self.judge.Compile()
    if res.status != RunResult.OK:
      ctx.errors.Error(self, "%s: Compile Error (%s)" % (self.judge.src_name, res.status))
      Console.PrintLog(self.judge.ReadCompileLog())
      return False
    return True

  def _CompileReferenceSolution(self, ctx):
    """
    Compile the reference solution.
    """
    reference_solution = self.problem.reference_solution
    if reference_solution is None:
      ctx.errors.Error(self, "Reference solution is not available")
      return False
    return reference_solution.Build(ctx)

  def _RunReferenceSolution(self, ctx):
    """
    Run the reference solution to generate reference outputs.
    """
    reference_solution = self.problem.reference_solution
    if reference_solution is None:
      ctx.errors.Error(self, "Reference solution is not available")
      return False
    Console.PrintAction("REFRUN", reference_solution, progress=True)
    infiles = self.ListInputFiles()
    for (i, infile) in enumerate(infiles):
      difffile = os.path.splitext(infile)[0] + FileNames.DIFF_EXT
      if os.path.isfile(os.path.join(self.out_dir, difffile)):
        continue
      Console.PrintAction(
        "REFRUN", reference_solution,
        "[%d/%d] %s" % (i+1, len(infiles), infile),
        progress=True)
      res = reference_solution.Run(
        args=[], cwd=self.out_dir,
        input=os.path.join(self.out_dir, infile),
        output=os.path.join(self.out_dir, difffile),
        timeout=None)
      if res.status != RunResult.OK:
        ctx.errors.Error(reference_solution, res.status)
        return False
    Console.PrintAction(
      "REFRUN", reference_solution)
    return True

  def _GenerateConcatTest(self, ctx):
    if not self.concat_test:
      return True
    Console.PrintAction("GENERATE", self, progress=True)
    concat_infile = FileNames.CONCAT_INFILE
    concat_difffile = FileNames.CONCAT_DIFFFILE
    FileUtil.CreateEmptyFile(os.path.join(self.out_dir, concat_infile))
    FileUtil.CreateEmptyFile(os.path.join(self.out_dir, concat_difffile))
    separator = FileUtil.ReadFile(self.separator_file)
    terminator = FileUtil.ReadFile(self.terminator_file)
    infiles = self.ListInputFiles()
    for (i, infile) in enumerate(infiles):
      infile = os.path.join(self.out_dir, infile)
      difffile = os.path.splitext(infile)[0] + FileNames.DIFF_EXT
      Console.PrintAction(
        "GENERATE", self,
        "[%d/%d] %s / %s" % (i+1, len(infiles),
                             concat_infile, concat_difffile),
        progress=True)
      in_content = FileUtil.ReadFile(infile)
      diff_content = FileUtil.ReadFile(difffile)
      if i > 0:
        FileUtil.AppendFile(separator, os.path.join(self.out_dir, concat_infile))
      FileUtil.AppendFile(in_content, os.path.join(self.out_dir, concat_infile))
      FileUtil.AppendFile(diff_content, os.path.join(self.out_dir, concat_difffile))
    FileUtil.AppendFile(terminator, os.path.join(self.out_dir, concat_infile))
    Console.PrintAction(
      "GENERATE", self,
      "%s (%d bytes) / %s (%d bytes)" % (
        concat_infile,
        os.path.getsize(os.path.join(self.out_dir, concat_infile)),
        concat_difffile,
        os.path.getsize(os.path.join(self.out_dir, concat_difffile)),
        ))
    return True

  def Test(self, ctx):
    """
    Test all solutions.
    """
    if not self.Build(ctx):
      return []
    results = []
    for solution in self.problem.solutions:
      results.extend(self.TestSolution(solution, ctx))
    return results

  def TestSolution(self, solution, ctx):
    """
    Test a single solution.
    """
    # Note: though Tests.Test() executes Tests.Build(), it is also
    # required here because Solution.Test() calls this function directly.
    if not self.Build(ctx):
      result = TestResult(self.problem, solution, [])
      result.good = False
      result.passed = False
      result.detail = "Failed to build tests"
      return [result]
    if not solution.Build(ctx):
      result = TestResult(self.problem, solution, [])
      result.good = False
      result.passed = False
      result.detail = "Compile Error"
      return [result]
    Console.PrintAction("TEST", solution, progress=True)
    if not solution.IsCorrect() and solution.challenge_cases:
      result = self._TestSolutionWithChallengeCases(solution, ctx)
    else:
      result = self._TestSolutionWithAllCases(solution, ctx)
    if result.good and result.passed:
      assert not result.detail
      if result.IsTimeStatsAvailable():
        result.detail = result.GetTimeStats()
      else:
        result.detail = "(no test)"
    else:
      assert result.detail
    status_row = []
    status_row += [
      result.good and Console.CYAN or Console.RED,
      result.passed and "PASSED" or "FAILED",
      Console.NORMAL,
      " ",
      result.detail]
    if result.cached:
      status_row += [" ", "(cached)"]
    Console.PrintAction("TEST", solution, *status_row)
    if solution.IsCorrect() and not result.good:
      assert result.ruling_file
      judgefile = os.path.splitext(result.ruling_file)[0] + FileNames.JUDGE_EXT
      log = FileUtil.ReadFile(os.path.join(solution.out_dir, judgefile))
      Console.PrintLog(log)
    return [result]

  def _TestSolutionWithChallengeCases(self, solution, ctx):
    """
    Test a wrong solution which has explicitly-specified challenge cases.
    """
    infiles = self.ListInputFiles()
    challenge_cases = self._SortInputFiles(solution.challenge_cases)
    cookie = solution.GetCacheStamp()
    result = TestResult(self.problem, solution, challenge_cases)
    # Ensure all challenge cases exist.
    all_exists = True
    for infile in challenge_cases:
      if infile not in infiles:
        ctx.errors.Error(solution,
                         "Challenge case not found: %s" % infile)
        all_exists = False
    if not all_exists:
      result.good = False
      result.passed = False
      result.detail = "Challenge case not found"
      return result
    # Try challenge cases.
    for (i, infile) in enumerate(challenge_cases):
      Console.PrintAction(
        "TEST", solution,
        "[%d/%d] %s" % (i+1, len(challenge_cases), infile),
        progress=True)
      (verdict, time, cached) = self._TestOneCase(
        solution, infile, cookie, ctx)
      if cached:
        result.cached = True
      result.cases[infile].verdict = verdict
      if verdict == TestResult.AC:
        result.ruling_file = infile
        result.good = False
        result.passed = True
        result.detail = "%s: Unexpectedly Accepted" % infile
        ctx.errors.Error(solution, result.detail, quiet=True)
        break
      elif verdict not in (TestResult.WA, TestResult.TLE, TestResult.RE):
        result.ruling_file = infile
        result.good = False
        result.passed = False
        result.detail = "%s: Judge Error" % infile
        ctx.errors.Error(solution, result.detail, quiet=True)
        break
    if result.good is None:
      result.good = True
      result.passed = False
      result.detail = "Expectedly Failed"
    return result

  def _TestSolutionWithAllCases(self, solution, ctx):
    """
    Test a solution without challenge cases.
    The solution can be marked as wrong but without challenge cases.
    """
    infiles = self.ListInputFiles(include_concat=True)
    cookie = solution.GetCacheStamp()
    result = TestResult(self.problem, solution, infiles)
    # Try all cases.
    for (i, infile) in enumerate(infiles):
      Console.PrintAction(
        "TEST", solution,
        "[%d/%d] %s" % (i+1, len(infiles), infile),
        progress=True)
      ignore_timeout = (infile == FileNames.CONCAT_INFILE)
      (verdict, time, cached) = self._TestOneCase(
        solution, infile, cookie, ctx, ignore_timeout=ignore_timeout)
      if cached:
        result.cached = True
      result.cases[infile].verdict = verdict
      if verdict not in (TestResult.AC, TestResult.WA, TestResult.TLE, TestResult.RE):
        result.ruling_file = infile
        result.good = False
        result.passed = False
        result.detail = "%s: Judge Error" % infile
        ctx.errors.Error(solution, result.detail, quiet=True)
        break
      elif verdict != TestResult.AC:
        result.ruling_file = infile
        result.passed = False
        result.detail = "%s: %s" % (infile, verdict)
        if solution.IsCorrect():
          result.good = False
          ctx.errors.Error(solution, result.detail, quiet=True)
        else:
          result.good = True
        break
      result.cases[infile].time = time
    if result.good is None:
      result.good = solution.IsCorrect()
      result.passed = True
      if not result.good:
        result.detail = "Unexpectedly Passed"
    return result

  def _TestOneCase(self, solution, infile, cookie, ctx, ignore_timeout=False):
    """
    Test a solution with one case.
    Cache results if option is set.
    Return (verdict, time, cached).
    """
    cachefile = os.path.join(
      solution.out_dir,
      os.path.splitext(infile)[0] + FileNames.CACHE_EXT)
    if ctx.options.cache_tests:
      if cookie is not None and os.path.isfile(cachefile):
        try:
          (cached_cookie, result) = FileUtil.PickleLoad(cachefile)
        except:
          ctx.errors.Exception(solution)
          cached_cookie = None
        if cached_cookie == cookie:
          return tuple(list(result)+[True])
    result = self._TestOneCaseNoCache(solution, infile, ignore_timeout=ignore_timeout)
    try:
      FileUtil.PickleSave((cookie, result), cachefile)
    except:
      ctx.errors.Exception(solution)
    return tuple(list(result)+[False])

  def _TestOneCaseNoCache(self, solution, infile, ignore_timeout=False):
    """
    Test a solution with one case.
    Never cache results.
    Return (verdict, time).
    """
    outfile = os.path.splitext(infile)[0] + FileNames.OUT_EXT
    difffile = os.path.splitext(infile)[0] + FileNames.DIFF_EXT
    judgefile = os.path.splitext(infile)[0] + FileNames.JUDGE_EXT
    timeout = self.problem.timeout
    if ignore_timeout:
      timeout = None
    res = solution.Run(
      args=[], cwd=solution.out_dir,
      input=os.path.join(self.out_dir, infile),
      output=os.path.join(solution.out_dir, outfile),
      timeout=timeout)
    if res.status == RunResult.TLE:
      return (TestResult.TLE, None)
    if res.status != RunResult.OK:
      return (TestResult.RE, None)
    time = res.time
    res = self.judge.Run(
      args=['--infile', os.path.join(self.out_dir, infile),
            '--difffile', os.path.join(self.out_dir, difffile),
            '--outfile', os.path.join(solution.out_dir, outfile)],
      cwd=self.out_dir,
      input=os.devnull,
      output=os.path.join(solution.out_dir, judgefile),
      timeout=None)
    if res.status == RunResult.OK:
      return (TestResult.AC, time)
    if res.status == RunResult.NG:
      return (TestResult.WA, None)
    return ("Validator " + res.status, None)

  def Pack(self, ctx):
    """
    Pack test cases.
    """
    if self.IsBuildCached():
      # TODO: do caching of packed tests output here.
      pass
    else:
      if not self.Build(ctx):
        return False
    infiles = self.ListInputFiles()
    Console.PrintAction("PACK", self, progress=True)
    if not os.path.isdir(self.pack_dir):
      try:
        FileUtil.MakeDir(self.pack_dir)
      except:
        ctx.errors.Exception(self)
        return False
    for (i, infile) in enumerate(infiles):
      basename = os.path.splitext(infile)[0]
      difffile = basename + FileNames.DIFF_EXT
      packed_infile = str(i+1) + FileNames.IN_EXT
      packed_difffile = str(i+1) + FileNames.DIFF_EXT
      try:
        Console.PrintAction(
          "PACK",
          self,
          "%s -> %s" % (infile, packed_infile),
          progress=True)
        FileUtil.CopyFile(os.path.join(self.out_dir, infile),
                          os.path.join(self.pack_dir, packed_infile))
        Console.PrintAction(
          "PACK",
          self,
          "%s -> %s" % (difffile, packed_difffile),
          progress=True)
        FileUtil.CopyFile(os.path.join(self.out_dir, difffile),
                          os.path.join(self.pack_dir, packed_difffile))
      except:
        ctx.errors.Exception(self)
        return False
    tar_args = ["tar", "czf",
                os.path.join(os.pardir, FileNames.TESTS_PACKED_TARBALL),
                os.curdir]
    Console.PrintAction(
      "PACK",
      self,
      " ".join(tar_args),
      progress=True)
    ret = -1
    try:
      devnull = FileUtil.OpenNull()
      ret = subprocess.call(tar_args,
                            cwd=self.pack_dir,
                            stdin=devnull,
                            stdout=devnull,
                            stderr=devnull)
    except:
      ctx.errors.Exception(self)
      return False
    if ret != 0:
      ctx.errors.Error(self, "tar failed: ret = %d" % ret)
      return False
    Console.PrintAction(
      "PACK",
      self,
      FileNames.TESTS_PACKED_TARBALL)
    return True

  def Clean(self, ctx):
    """
    Remove test cases.
    """
    Console.PrintAction("CLEAN", self)
    try:
      FileUtil.RemoveTree(self.out_dir)
    except:
      ctx.errors.Exception(self)

  def ListInputFiles(self, include_concat=False):
    """
    Enumerate input files.
    """
    infiles = []
    for infile in FileUtil.ListDir(self.out_dir, True):
      if not infile.endswith(FileNames.IN_EXT):
        continue
      if not os.path.isfile(os.path.join(self.out_dir, infile)):
        continue
      infiles.append(infile)
    infiles = self._SortInputFiles(infiles)
    if include_concat:
      if os.path.isfile(os.path.join(self.out_dir, FileNames.CONCAT_INFILE)):
        infiles.append(FileNames.CONCAT_INFILE)
    return infiles

  def _SortInputFiles(self, infiles):
    """
    Compare input file names in a little bit smart way.
    """
    infiles = infiles[:]
    def tokenize_cmp(a, b):
      def tokenize(s):
        def replace_digits(match):
          return "%08s" % match.group(0)
        return re.sub(r'\d+', replace_digits, s)
      return cmp(tokenize(a), tokenize(b))
    infiles.sort(tokenize_cmp)
    return infiles



class Solution(BuildableObject):
  """
  Represents a single solution.
  """

  CONFIG_FILE = FileNames.SOLUTION_FILE

  def __init__(self, name, base_dir, parent):
    super(Solution, self).__init__(
      name, base_dir, os.path.join(parent.out_dir, name), parent)
    self.problem = parent

  def _PreLoad(self, ctx):
    self.code = None
    self._AddCodeRegisterer('code', 'solution')
    self._export_dict["solution"] = self
    self._export_dict["root"] = self.root
    self._export_dict["problem"] = self.problem

  def _PostLoad(self, ctx):
    source_exts = {
      '.c': self.c_solution,
      '.cc': self.cxx_solution,
      '.cpp': self.cxx_solution,
      '.java': self.java_solution,
      }
    # If the code is not explicitly specified, guess it.
    if self.code is None:
      src = None
      solution_func = None
      ambiguous = False
      for name in FileUtil.ListDir(self.src_dir):
        if not os.path.isfile(os.path.join(self.src_dir, name)):
          continue
        ext = os.path.splitext(name)[1]
        if ext in source_exts:
          if src is not None:
            ambiguous = True
            break
          src = name
          solution_func = source_exts[ext]
      if ambiguous:
        ctx.errors.Error(self,
                         ("Multiple source files found; " +
                          "specify explicitly in " +
                          self.CONFIG_FILE))
      elif src is None:
        ctx.errors.Error(self, "Source file not found")
      else:
        solution_func(src=src)
    # Decide if this solution is correct or not.
    if 'CHALLENGE_CASES' in self.config:
      self.correct = False
      self.challenge_cases = self.config['CHALLENGE_CASES']
    else:
      self.correct = True
      self.challenge_cases = None

  def IsCorrect(self):
    """
    Returns whether this is correct solution.
    """
    return self.correct

  def Build(self, ctx):
    """
    Build this solution.
    """
    #Console.PrintAction("BUILD", self)
    if self.IsBuildCached():
      Console.PrintAction("COMPILE", self, "up-to-date")
      return True
    if not self.code.QUIET_COMPILE:
      Console.PrintAction("COMPILE", self)
    res = self.code.Compile()
    log = self.code.ReadCompileLog()
    if res.status != RunResult.OK:
      ctx.errors.Error(self, "Compile Error (%s)" % res.status)
      Console.PrintLog(log)
      return False
    if log:
      ctx.errors.Warning(self, "Compiler warnings found")
      Console.PrintLog(log)
    if not self.SetCacheStamp(ctx):
      return False
    return True

  def Test(self, ctx):
    """
    Test this solution.
    """
    return self.problem.tests.TestSolution(self, ctx)

  def Run(self, args, cwd, input, output, timeout):
    """
    Run this solution.
    """
    return self.code.Run(args=args, cwd=cwd,
                         input=input, output=output, timeout=timeout)

  def Clean(self, ctx):
    """
    Clean this solution.
    """
    Console.PrintAction("CLEAN", self)
    e = self.code.Clean()
    if e:
      ctx.errors.Exception(self, e)
      return False
    return True



class Rime(object):
  """
  The main class of Rime.
  """

  def Main(self, args):
    """
    Main method called when invoked as stand-alone script.
    """
    # Banner.
    Console.Print("Rime: Tool for Programming Contest Organizers")
    Console.Print()
    # Parse arguments.
    (cmd, params, options) = self._ParseArgs(args)
    if cmd is None or options.show_help:
      self.PrintHelp()
      return 0
    ctx = RimeContext(options)
    # Check system.
    if not self._CheckSystem(ctx):
      return 1
    # Try to load config files.
    root = self.LoadRoot(os.getcwd(), ctx)
    if not root:
      Console.PrintError("RIMEROOT not found. Make sure you are in Rime subtree.")
      return 1
    if ctx.errors.HasError():
      Console.PrintError("Encountered error on loading config files.")
      return 1
    # Decide target object.
    # Note: currently all commands recognizes first parameter as base_dir.
    if params:
      base_dir = os.path.abspath(params[0])
      params = params[1:]
    else:
      base_dir = os.getcwd()
    obj = root.FindByBaseDir(base_dir)
    if not obj:
      Console.PrintError("Target directory is not managed by Rime.")
      return 1
    # Call.
    if cmd == 'build':
      success = obj.Build(ctx)
      Console.Print("Finished Build.")
      Console.Print()
    elif cmd == 'test':
      results = obj.Test(ctx)
      Console.Print("Finished Test.")
      Console.Print()
      self.PrintTestSummary(results)
    elif cmd == 'clean':
      success = obj.Clean(ctx)
      Console.Print("Finished Clean.")
      Console.Print()
    elif cmd == 'pack':
      success = obj.Pack(ctx)
      Console.Print("Finished Pack.")
      Console.Print()
    else:
      Console.PrintError("Unknown command: %s" % cmd)
      return 1
    Console.Print()
    Console.Print(Console.BOLD, "Error Summary:", Console.NORMAL)
    ctx.errors.PrintSummary()
    return 0

  def LoadRoot(self, cwd, ctx):
    """
    Load configs and return RimeRoot instance.
    Location of root directory is searched upward from cwd.
    If RIMEROOT cannot be found, return None.
    """
    path = cwd
    while not RimeRoot.CanLoadFrom(path):
      (head, tail) = os.path.split(path)
      if head == path:
        return None
      path = head
    root = RimeRoot(None, path, None)
    root.Load(ctx)
    return root

  def PrintHelp(self):
    """
    Just print help message.
    """
    print HELP_MESSAGE

  def PrintTestSummary(self, results):
    if len(results) == 0:
      return
    Console.Print(Console.BOLD, "Test Summary:", Console.NORMAL)
    solution_name_width = max(
      map(lambda t: len(t.solution.name), results))
    last_problem = None
    for result in sorted(results, TestResult.CompareForListing):
      if last_problem is not result.problem:
        problem_row = [Console.BOLD,
                       Console.CYAN,
                       result.problem.name,
                       Console.NORMAL,
                       " ... %d solutions, %d tests" %
                       (len(result.problem.solutions),
                        len(result.problem.tests.ListInputFiles()))]
        Console.Print(*problem_row)
        last_problem = result.problem
      status_row = ["  "]
      status_row += [
        result.solution.IsCorrect() and Console.GREEN or Console.YELLOW,
        result.solution.name.ljust(solution_name_width),
        Console.NORMAL,
        " "]
      status_row += [
        result.good and Console.CYAN or Console.RED,
        result.passed and "PASSED" or "FAILED",
        Console.NORMAL,
        " "]
      if result.good:
        if result.passed:
          if result.IsTimeStatsAvailable():
            status_row += [result.GetTimeStats()]
          else:
            status_row += ["(no test)"]
        else:
          status_row += ["Expectedly Failed"]
      else:
        if result.passed:
          status_row += ["Unexpectedly Passed"]
        else:
          if result.ruling_file:
            status_row += [result.cases[result.ruling_file].verdict,
                           ": ",
                           result.ruling_file]
          else:
            status_row += [result.detail]
      if result.cached:
        status_row += [" ", "(cached)"]
      Console.Print(*status_row)

  def GetOptionParser(self):
    """
    Construct optparse.OptionParser object for Rime.
    """
    parser = optparse.OptionParser(add_help_option=False)
    parser.add_option('-h', '--help', dest='show_help',
                      default=False, action="store_true")
    parser.add_option('-C', '--cache-tests', dest='cache_tests',
                      default=False, action="store_true")
    parser.add_option('-d', '--debug', dest='debug',
                      default=0, action="count")
    return parser

  def GetDefaultOptions(self):
    """
    Get default options object.
    """
    return self.GetOptionParser().get_default_values()

  def _ParseArgs(self, args):
    """
    Parse args and return (cmd, params, options) tuple.
    """
    parser = self.GetOptionParser()
    (options, args) = parser.parse_args(args[1:])
    cmd = args[0].lower() if args else None
    if len(args) >= 2:
      params = args[1:]
    else:
      params = []
    return (cmd, params, options)

  def _CheckSystem(self, ctx):
    """
    Check the sytem environment.
    """
    system = platform.system()
    if system in ('Windows', 'Microsoft') or system.startswith('CYGWIN'):
      Console.Print("Note: Running Rime under Windows will be unstable.")
    return True


def main():
  try:
    # Instanciate Rime class and call Main().
    rime = Rime()
    ret = rime.Main(sys.argv)
    sys.exit(ret)
  except SystemExit:
    raise
  except KeyboardInterrupt:
    # Suppress stack trace when interrupted by Ctrl-C
    sys.exit(1)
  except Exception:
    # Print stack trace for debug.
    exc = sys.exc_info()
    sys.excepthook(*exc)
    sys.exit(1)


if __name__ == '__main__':
  main()

