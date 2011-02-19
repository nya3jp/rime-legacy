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
import functools
import inspect
import itertools
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
  -i, --ignore-errors  don't skip tests on failure
  -j, --jobs=num       run processes in parallel
  -p, --precise        don't run timing tasks concurrently
  -C, --cache-tests    cache test results [experimental]
  -d, --debug          print debug messages
  -h, --help           show this help message and exit
"""


## Utilities

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
      cls.Print(line)

# Call Init() on load time.
Console.Init()


## TaskGraph

# State of tasks.
RUNNING, WAITING, BLOCKED, READY, FINISHED, ABORTED = range(6)


class TaskBranch(object):

  def __init__(self, tasks, unsafe_interrupt=False):
    self.tasks = tasks
    self.interrupt = unsafe_interrupt


class TaskReturn(object):

  def __init__(self, value):
    self.value = value


class TaskBlock(object):

  pass


class _TaskRaise(object):
  """
  Internal only; do not return an instance of this class from generators.
  """

  def __init__(self, type, value=None, traceback=None):
    self.exc_info = (type, value, traceback)


class Bailout(Exception):

  def __init__(self, value=None):
    self.value = value


class TaskInterrupted(Exception):

  pass


class Task(object):

  def __hash__(self):
    """
    Hash function of Task. Usually users should override CacheKey() only.
    """
    if self.CacheKey() is None:
      return id(self)
    return hash(self.CacheKey())

  def __eq__(self, other):
    """
    Equality function of Task. Usually users should override CacheKey() only.
    """
    if not isinstance(other, Task):
      return False
    if self.CacheKey() is None and other.CacheKey() is None:
      return id(self) == id(other)
    return self.CacheKey() == other.CacheKey()

  def IsCacheable(self):
    """
    Checks if this task is cachable. Usually users should override CacheKey() only.
    """
    return self.CacheKey() is not None

  def IsExclusive(self):
    """
    Checks if this task is exclusive.

    If a task is exclusive, it runs only when no other task is blocked.
    """
    return False

  def CacheKey(self):
    """
    Returns the cache key of this task.

    Need to be overridden in subclasses. If this returns None, the task value is
    never cached.
    """
    raise NotImplementedError()

  def Continue(self, value=None):
    """
    Continues the task.

    Implementations can return these type of values:
    - TaskBranch: a list of tasks to be invoked next.
    - TaskReturn: a value to be returned to the caller.
    - TaskBlock: indicates this operation will block.
    - Task: treated as TaskBranch(task).
    - any other value: treated as TaskReturn(value).
    In addition to these, it can raise an exception, including Bailout.

    First invocation of this function will be with no parameter or None. If it
    returns TaskBranch, next parameter will be a list of the results of the
    specified tasks.
    """
    raise NotImplementedError()

  def Throw(self, type, value=None, traceback=None):
    """
    Throws in an exception.

    After Continue() or Throw() returned TaskBranch, if some of the branches
    raised an exception, this function is called. Return value of this
    function is treated in the same way as Continue().
    """
    raise NotImplementedError()

  def Poll(self):
    """
    Polls the blocked task.

    If the operation is ready, return True. This function should return
    immediately, and should not raise an exception.
    """
    return True

  def Wait(self):
    """
    Polls the blocked task.

    This function should wait until the operation gets ready. This function
    should not raise an exception.
    """
    pass

  def Close(self):
    """
    Closes the task.

    This is called once after Continue() or Throw() returned TaskReturn,
    they raised an exception, or the task was interrupted.
    The task should release all resources associated with it, such as
    running generators or opened processes.
    If this function raises an exception, the value returned by Continue()
    or Throw() is discarded.
    """
    pass


class GeneratorTask(Task):

  def __init__(self, it, key):
    self.it = it
    self.key = key

  def __repr__(self):
    return repr(self.key)

  def CacheKey(self):
    return self.key

  def Continue(self, value=None):
    try:
      return self.it.send(value)
    except StopIteration:
      return TaskReturn(None)

  def Throw(self, type, value=None, traceback=None):
    try:
      return self.it.throw(type, value, traceback)
    except StopIteration:
      return TaskReturn(None)

  def Close(self):
    try:
      self.it.close()
    except RuntimeError:
      # Python2.5 raises RuntimeError when GeneratorExit is ignored. This often
      # happens when yielding a return value from inside of try block, or even
      # Ctrl+C was pressed when in try block.
      pass

  @staticmethod
  def FromFunction(func):
    @functools.wraps(func)
    def MakeTask(*args, **kwargs):
      key = GeneratorTask._MakeCacheKey(func, args, kwargs)
      try:
        hash(key)
      except TypeError:
        raise ValueError(
          'Unhashable argument was passed to GeneratorTask function')
      it = func(*args, **kwargs)
      return GeneratorTask(it, key)
    return MakeTask

  @staticmethod
  def _MakeCacheKey(func, args, kwargs):
    return ('GeneratorTask', func, tuple(args), tuple(kwargs.items()))


class ExternalProcessTask(Task):

  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs
    self.proc = None
    if 'timeout' in kwargs:
      self.timeout = kwargs['timeout']
      del kwargs['timeout']
    else:
      self.timeout = None
    if 'exclusive' in kwargs:
      self.exclusive = kwargs['exclusive']
      del kwargs['exclusive']
    else:
      self.exclusive = False
    self.timer = None

  def CacheKey(self):
    # Never cache.
    return None

  def IsExclusive(self):
    return self.exclusive

  def Continue(self, value=None):
    if self.exclusive:
      return self._ContinueExclusive()
    else:
      return self._ContinueNonExclusive()

  def _ContinueExclusive(self):
    assert self.proc is None
    self._StartProcess()
    self.proc.wait()
    return TaskReturn(self._EndProcess())

  def _ContinueNonExclusive(self):
    if self.proc is None:
      self._StartProcess()
      return TaskBlock()
    elif not self.Poll():
      return TaskBlock()
    else:
      return TaskReturn(self._EndProcess())

  def Poll(self):
    assert self.proc is not None
    return self.proc.poll() is not None

  def Wait(self):
    assert self.proc is not None
    self.proc.wait()

  def Close(self):
    if self.timer is not None:
      self.timer.cancel()
      self.timer = None
    if self.proc is not None:
      try:
        os.kill(self.proc.pid, signal.SIGKILL)
      except:
        pass
      self.proc.wait()
      self.proc = None

  def _StartProcess(self):
    self.start_time = time.time()
    self.proc = subprocess.Popen(*self.args, **self.kwargs)
    if self.timeout is not None:
      def TimeoutKiller():
        try:
          os.kill(self.proc.pid, signal.SIGXCPU)
        except:
          pass
      self.timer = threading.Timer(self.timeout, TimeoutKiller)
      self.timer.start()
    else:
      self.timer = None

  def _EndProcess(self):
    self.end_time = time.time()
    self.time = self.end_time - self.start_time
    if self.timer is not None:
      self.timer.cancel()
      self.timer = None
    # Don't keep proc in cache.
    proc = self.proc
    self.proc = None
    return proc


class SerialTaskGraph(object):
  """
  TaskGraph which emulates normal serialized execution.
  """

  def __init__(self):
    self.cache = dict()

  def Close(self):
    pass

  def Run(self, task):
    if task not in self.cache:
      self.cache[task] = None
      value = (True, None)
      while True:
        try:
          if value[0]:
            result = task.Continue(value[1])
          elif isinstance(value[1][1], Bailout):
            result = task.Continue(value[1][1].value)
          else:
            result = task.Throw(*value[1])
        except StopIteration:
          result = TaskReturn(None)
        except:
          result = _TaskRaise(*sys.exc_info())
        if isinstance(result, TaskBranch):
          try:
            value = (True, [self.Run(subtask) for subtask in result.tasks])
          except:
            value = (False, sys.exc_info())
        elif isinstance(result, Task):
          try:
            value = (True, self.Run(result))
          except:
            value = (False, sys.exc_info())
        elif isinstance(result, TaskBlock):
          value = (True, None)
          task.Wait()
        elif isinstance(result, _TaskRaise):
          self.cache[task] = (False, result.exc_info)
          break
        elif isinstance(result, TaskReturn):
          self.cache[task] = (True, result.value)
          break
        else:
          self.cache[task] = (True, result)
          break
      try:
        task.Close()
      except:
        self.cache[task] = (False, sys.exc_info())
    if self.cache[task] is None:
      raise RuntimeException('Cyclic task dependency found')
    success, value = self.cache[task]
    if success:
      return value
    else:
      raise value[0], value[1], value[2]


class FiberTaskGraph(object):
  """
  TaskGraph which executes tasks with fibers (microthreads).

  FiberTaskGraph allows some tasks to be in blocked state in the same time.
  Branched tasks are executed in arbitrary order.
  """

  def __init__(self, parallelism, debug=0):
    self.parallelism = parallelism
    self.debug = debug
    self.cache = dict()
    self.task_graph = dict()
    self.task_interrupt = dict()
    self.task_counters = dict()
    self.task_waits = dict()
    self.task_state = dict()
    self.ready_tasks = []
    self.blocked_tasks = []
    self.pending_stack = []

  def Close(self):
    for task in self.task_state:
      if self.task_state[task] not in (FINISHED, ABORTED):
        self._InterruptTask(task)

  def Run(self, init_task):
    self.first_tick = time.clock()
    self.last_tick = self.first_tick
    self.cumulative_parallelism = 0.0
    self._BranchTask(None, [init_task])
    while self._RunNextTask():
      pass
    self._UpdateCumulativeParallelism()
    if self.last_tick > self.first_tick:
      parallelism_efficiency = (
        self.cumulative_parallelism /
        (self.parallelism * (self.last_tick - self.first_tick)))
    else:
      parallelism_efficiency = 1.0
    self._Log("Parallelism efficiency: %.2f%%" %
              (100.0 * parallelism_efficiency),
              level=1)
    assert self.task_state[None] == READY
    del self.task_state[None]
    del self.task_graph[None]
    success, value = self.cache[init_task]
    if success:
      return value
    elif isinstance(value, Bailout):
      return value.value
    else:
      raise value[0], value[1], value[2]

  def _RunNextTask(self):
    while len(self.ready_tasks) == 0:
      if not self._VisitBranch():
        self._WaitBlockedTasks()
    next_task = self.ready_tasks.pop(0)
    self._LogTaskStats()
    if next_task is None:
      return False
    if self.task_state[next_task] != READY:
      # Interrupted.
      return True
    exc_info = None
    if next_task in self.task_graph:
      if isinstance(self.task_graph[next_task], list):
        value = []
        for task in self.task_graph[next_task]:
          if task in self.cache:
            success, cached = self.cache[task]
            if success:
              value.append(cached)
            elif exc_info is None or isinstance(exc_info[1], TaskInterrupted):
              exc_info = cached
      else:
        success, cached = self.cache[self.task_graph[next_task]]
        if success:
          value = cached
        else:
          exc_info = cached
      del self.task_graph[next_task]
    else:
      value = None
    self._SetTaskState(next_task, RUNNING)
    if exc_info is not None:
      if isinstance(exc_info[1], Bailout):
        self._ContinueTask(next_task, exc_info[1].value)
      else:
        self._ThrowTask(next_task, exc_info)
    else:
      self._ContinueTask(next_task, value)
    return True

  def _VisitBranch(self):
    if not self.pending_stack:
      return False
    # Visit branches by depth first.
    task, subtask = self.pending_stack.pop()
    self._BeginTask(subtask, task)
    return True

  def _ContinueTask(self, task, value):
    assert self.task_state[task] == RUNNING
    assert not task.IsExclusive() or len(self.blocked_tasks) == 0
    self._LogDebug('_ContinueTask: %s: entering' % task)
    try:
      result = task.Continue(value)
    except:
      self._LogDebug('_ContinueTask: %s: exception raised' % task)
      self._ProcessTaskException(task, sys.exc_info())
    else:
      self._LogDebug('_ContinueTask: %s: exited' % task)
      self._ProcessTaskResult(task, result)

  def _ThrowTask(self, task, exc_info):
    assert self.task_state[task] == RUNNING
    assert not task.IsExclusive() or len(self.blocked_tasks) == 0
    self._LogDebug('_ThrowTask: %s: entering' % task)
    try:
      result = task.Throw(*exc_info)
    except:
      self._LogDebug('_ThrowTask: %s: exception raised' % task)
      self._ProcessTaskException(task, sys.exc_info())
    else:
      self._LogDebug('_ThrowTask: %s: exited' % task)
      self._ProcessTaskResult(task, result)

  def _ProcessTaskResult(self, task, result):
    assert self.task_state[task] == RUNNING
    if isinstance(result, Task):
      self._LogDebug('_ProcessTaskResult: %s: received Task' % task)
      self._BranchTask(task, result)
    elif isinstance(result, TaskBranch):
      self._LogDebug('_ProcessTaskResult: %s: received TaskBranch '
                     'with %d tasks' % (task, len(result.tasks)))
      self._BranchTask(task, list(result.tasks), result.interrupt)
    elif isinstance(result, TaskReturn):
      self._LogDebug('_ProcessTaskResult: %s: received TaskReturn' % task)
      self._FinishTask(task, result.value)
    elif isinstance(result, TaskBlock):
      self._LogDebug('_ProcessTaskResult: %s: received TaskBlock' % task)
      self._BlockTask(task)
    else:
      self._LogDebug('_ProcessTaskResult: %s: received unknown type,'
                     'implying TaskReturn' % task)
      self._FinishTask(task, result)

  def _ProcessTaskException(self, task, exc_info):
    assert self.task_state[task] == RUNNING
    try:
      task.Close()
    except:
      # Ignore the exception.
      pass
    self._ExceptTask(task, exc_info)

  def _BranchTask(self, task, subtasks, interrupt=False):
    assert task is None or self.task_state[task] == RUNNING
    self.task_graph[task] = subtasks
    if not isinstance(subtasks, list):
      assert isinstance(subtasks, Task)
      subtasks = [subtasks]
    if len(subtasks) == 0:
      self._LogDebug('_BranchTask: %s: zero branch, fast return' % task)
      self.ready_tasks.insert(0, task)
      self._SetTaskState(task, READY)
      self._LogTaskStats()
      return
    self.task_interrupt[task] = interrupt
    self.task_counters[task] = len(subtasks)
    # The branches are half-expanded, but don't complete the operation here
    # so that too many branches are opened.
    for subtask in reversed(subtasks):
      self.pending_stack.append((task, subtask))
    self._SetTaskState(task, WAITING)

  def _BeginTask(self, task, parent_task):
    if task in self.cache:
      assert self.task_state[task] in (FINISHED, ABORTED)
      self._LogDebug('_BeginTask: %s: cache hit' % task)
      success = self.cache[task][0]
      if success:
        self._ResolveTask(parent_task)
      else:
        self._BailoutTask(parent_task)
    elif parent_task not in self.task_counters:
      # Some sibling task already bailed out. Skip this task.
      self._LogDebug('_BeginTask: %s: sibling task bailed out' % task)
      return
    else:
      if task in self.task_waits:
        assert self.task_state[task] in (WAITING, BLOCKED)
        self._LogDebug('_BeginTask: %s: running' % task)
        self.task_waits[task].append(parent_task)
      else:
        assert task not in self.task_state
        self._LogDebug('_BeginTask: %s: starting' % task)
        self.task_waits[task] = [parent_task]
        self._SetTaskState(task, RUNNING)
        if task.IsExclusive():
          self._WaitBlockedTasksUntilEmpty()
        self._ContinueTask(task, None)

  def _FinishTask(self, task, value):
    assert self.task_state[task] == RUNNING
    try:
      task.Close()
    except:
      self._ExceptTask(task, sys.exc_info())
      return
    self.cache[task] = (True, value)
    self._LogDebug('_FinishTask: %s: finished, returned: %s' % (task, value))
    for wait_task in self.task_waits[task]:
      self._ResolveTask(wait_task)
    del self.task_waits[task]
    self._SetTaskState(task, FINISHED)

  def _ExceptTask(self, task, exc_info):
    assert self.task_state[task] in (RUNNING, BLOCKED)
    assert task not in self.cache
    self.cache[task] = (False, exc_info)
    self._LogDebug('_ExceptTask: %s: exception raised: %s' %
                   (task, exc_info[0].__name__))
    bailouts = self.task_waits[task]
    del self.task_waits[task]
    if self.task_state[task] == BLOCKED:
      del self.task_counters[task]
    self._SetTaskState(task, ABORTED)
    for bailout in bailouts:
      self._BailoutTask(bailout)

  def _BlockTask(self, task):
    assert self.task_state[task] == RUNNING
    assert len(self.blocked_tasks) < self.parallelism
    self.task_counters[task] = 1
    self._UpdateCumulativeParallelism()
    self.blocked_tasks.insert(0, task)
    self._SetTaskState(task, BLOCKED)
    self._LogTaskStats()
    self._LogDebug('_BlockTask: %s: pushed to blocked_tasks' % task)
    self._WaitBlockedTasksUntilNotFull()
    assert len(self.blocked_tasks) < self.parallelism

  def _WaitBlockedTasksUntilEmpty(self):
    self._LogDebug('_WaitBlockedTasksUntilEmpty: %d blocked tasks' %
                   len(self.blocked_tasks))
    while len(self.blocked_tasks) > 0:
      self._WaitBlockedTasks()

  def _WaitBlockedTasksUntilNotFull(self):
    self._LogDebug('_WaitBlockedTasksUntilNotFull: %d blocked tasks' %
                   len(self.blocked_tasks))
    if len(self.blocked_tasks) == self.parallelism:
      self._Log('Maximum parallelism reached, waiting for blocked tasks',
                level=2)
      self._WaitBlockedTasks()
      self._Log('Blocked task ready (%d -> %d)' %
                (self.parallelism, len(self.blocked_tasks)),
                level=2)

  def _WaitBlockedTasks(self):
    assert len(self.blocked_tasks) > 0
    self._LogTaskStats()
    self._LogDebug('_WaitBlockedTasks: waiting')
    while True:
      resolved = self._PollBlockedTasks()
      if resolved > 0:
        break
      self._Sleep()
    self._LogDebug('_WaitBlockedTasks: resolved %d blocked tasks' % resolved)

  def _PollBlockedTasks(self):
    i = 0
    resolved = 0
    while i < len(self.blocked_tasks):
      task = self.blocked_tasks[i]
      assert self.task_state[task] == BLOCKED
      success = task.Poll()
      if success:
        self._ResolveTask(task)
        resolved += 1
        self._UpdateCumulativeParallelism()
        self.blocked_tasks.pop(i)
        self._LogTaskStats()
      else:
        i += 1
    return resolved

  def _ResolveTask(self, task):
    if task not in self.task_counters:
      self._LogDebug('_ResolveTask: %s: resolved, but already bailed out' % task)
      return
    assert self.task_state[task] in (WAITING, BLOCKED)
    self._LogDebug('_ResolveTask: %s: resolved, counter: %d -> %d' %
                   (task, self.task_counters[task], self.task_counters[task]-1))
    self.task_counters[task] -= 1
    if self.task_counters[task] == 0:
      if task in self.task_graph and isinstance(self.task_graph[task], list):
        # Multiple branches.
        self.ready_tasks.append(task)
      else:
        # Serial execution or blocked task.
        self.ready_tasks.insert(0, task)
      if task in self.task_interrupt:
        del self.task_interrupt[task]
      del self.task_counters[task]
      self._SetTaskState(task, READY)
      self._LogDebug('_ResolveTask: %s: pushed to ready_task' % task)
      self._LogTaskStats()

  def _BailoutTask(self, task):
    if task not in self.task_counters:
      self._LogDebug('_BailoutTask: %s: multiple bail out' % task)
      return
    assert self.task_state[task] in (WAITING, BLOCKED)
    self._LogDebug('_BailoutTask: %s: bailing out' % task)
    if task in self.task_graph and isinstance(self.task_graph[task], list):
      # Multiple branches.
      self.ready_tasks.append(task)
    else:
      # Serial execution or blocked task.
      self.ready_tasks.insert(0, task)
    interrupt = False
    if task in self.task_interrupt:
      interrupt = self.task_interrupt[task]
      del self.task_interrupt[task]
    del self.task_counters[task]
    self._SetTaskState(task, READY)
    self._LogDebug('_BailoutTask: %s: pushed to ready_task' % task)
    if interrupt and task in self.task_graph:
      for subtask in self.task_graph[task]:
        self._InterruptTask(subtask)

  def _InterruptTask(self, task):
    if (task is None or task not in self.task_state or
        self.task_state[task] not in (WAITING, BLOCKED, READY)):
      return
    self._LogDebug('_InterruptTask: %s: interrupted' % task)
    try:
      task.Close()
    except:
      pass
    # Simulate as if the task raised an exception.
    subtasks = []
    if task in self.task_graph:
      subtasks = self.task_graph[task]
      del self.task_graph[task]
      if not isinstance(subtasks, list):
        subtasks = [subtasks]
    if task in self.task_interrupt:
      del self.task_interrupt[task]
    if task in self.task_counters:
      del self.task_counters[task]
    if self.task_state[task] == BLOCKED:
      self._UpdateCumulativeParallelism()
      self.blocked_tasks.remove(task)
    self._SetTaskState(task, RUNNING)
    self._ExceptTask(task, (TaskInterrupted, TaskInterrupted(), None))
    for subtask in subtasks:
      self._InterruptTask(subtask)

  def _UpdateCumulativeParallelism(self):
    cur_tick = time.clock()
    self.cumulative_parallelism += (
      (cur_tick - self.last_tick) * len(self.blocked_tasks))
    self.last_tick = cur_tick

  def _Sleep(self):
    time.sleep(0.01)

  def _SetTaskState(self, task, state):
    if state == RUNNING:
      assert task not in self.cache
      assert task not in self.task_graph
      assert task not in self.task_interrupt
      assert task not in self.task_counters
      assert task is None or task in self.task_waits
    elif state == WAITING:
      assert task not in self.cache
      assert task in self.task_graph
      assert task in self.task_interrupt
      assert task in self.task_counters
      assert task is None or task in self.task_waits
    elif state == BLOCKED:
      assert task not in self.cache
      assert task not in self.task_graph
      assert task not in self.task_interrupt
      assert self.task_counters.get(task) == 1
      assert task in self.task_waits
    elif state == READY:
      assert task not in self.cache
      assert task not in self.task_interrupt
      assert task not in self.task_counters
      assert task is None or task in self.task_waits
    elif state == FINISHED:
      assert task in self.cache and self.cache[task][0]
      assert task not in self.task_graph
      assert task not in self.task_interrupt
      assert task not in self.task_counters
      assert task not in self.task_waits
    elif state == ABORTED:
      assert task in self.cache and not self.cache[task][0]
      assert task not in self.task_graph
      assert task not in self.task_interrupt
      assert task not in self.task_counters
      assert task not in self.task_waits
    else:
      raise AssertionError('Unknown state: ' + str(state))
    self.task_state[task] = state

  def _LogTaskStats(self):
    stats = [0] * 6
    for state in self.task_state.values():
      stats[state] += 1
    self._LogDebug(('RUNNING %d, WAITING %d, BLOCKED %d, '
                    'READY %d, FINISHED %d, ABORTED %d') % tuple(stats))

  def _Log(self, msg, level):
    if self.debug >= level:
      Console.Print(msg)

  def _LogDebug(self, msg):
    self._Log(msg, level=3)


## Rime objects

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

  def IsTimeStatsAvailable(self, ctx):
    """
    Checks if time statistics are available.
    """
    return ((ctx.options.precise or ctx.options.parallelism == 1) and
            self.files and
            all([c.verdict == TestResult.AC for c in self.cases.values()]))

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

  def Run(self, args, cwd, input, output, timeout, precise, redirect_error=False):
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

  @GeneratorTask.FromFunction
  def Compile(self):
    """
    Compile the code and return (RunResult, log) pair.
    """
    try:
      for name in self.prereqs:
        if not FileUtil.LocateBinary(name):
          yield RunResult("%s: %s" % (RunResult.PM, name), None)
      self.MakeOutDir()
      result = yield self._ExecForCompile(args=self.compile_args)
    except Exception, e:
      result = RunResult(str(e), None)
    yield result

  @GeneratorTask.FromFunction
  def Run(self, args, cwd, input, output, timeout, precise, redirect_error=False):
    """
    Run the code and return RunResult.
    """
    try:
      result = yield self._ExecForRun(
        args=self.run_args+args, cwd=cwd,
        input=input, output=output, timeout=timeout, precise=precise,
        redirect_error=redirect_error)
    except Exception, e:
      result = RunResult(str(e), None)
    yield result

  @GeneratorTask.FromFunction
  def Clean(self):
    """
    Clean the output directory.
    Return an exception object on error.
    """
    try:
      FileUtil.RemoveTree(self.out_dir)
    except Exception, e:
      yield e
    else:
      yield None

  def ReadCompileLog(self):
    return FileUtil.ReadFile(os.path.join(self.out_dir, self.log_name))

  @GeneratorTask.FromFunction
  def _ExecForCompile(self, args):
    with open(os.path.join(self.out_dir, self.log_name), 'w') as outfile:
      yield (yield self._ExecInternal(
          args=args, cwd=self.src_dir,
          stdin=FileUtil.OpenNull(), stdout=outfile, stderr=subprocess.STDOUT))

  @GeneratorTask.FromFunction
  def _ExecForRun(self, args, cwd, input, output, timeout, precise,
                  redirect_error=False):
    with open(input, 'r') as infile:
      with open(output, 'w') as outfile:
        if redirect_error:
          errfile = subprocess.STDOUT
        else:
          errfile = FileUtil.OpenNull()
        yield (yield self._ExecInternal(
            args=args, cwd=cwd,
            stdin=infile, stdout=outfile, stderr=errfile, timeout=timeout,
            precise=precise))

  @GeneratorTask.FromFunction
  def _ExecInternal(self, args, cwd, stdin, stdout, stderr,
                    timeout=None, precise=False):
    task = ExternalProcessTask(
      args, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, timeout=timeout,
      exclusive=precise)
    proc = yield task
    code = proc.returncode
    # Retry if TLE.
    if not precise and code == -(signal.SIGXCPU):
      self._ResetIO(stdin, stdout, stderr)
      task = ExternalProcessTask(
        args, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, timeout=timeout,
        exclusive=True)
      proc = yield task
      code = proc.returncode
    if code == 0:
      status = RunResult.OK
    elif code == -(signal.SIGXCPU):
      status = RunResult.TLE
    elif code < 0:
      status = RunResult.RE
    else:
      status = RunResult.NG
    yield RunResult(status, task.time)

  def _ResetIO(self, *args):
    for f in args:
      if f is None:
        continue
      try:
        f.seek(0)
        f.truncate()
      except IOError:
        pass


class CCode(FileBasedCode):

  def __init__(self, src_name, src_dir, out_dir, flags):
    super(CCode, self).__init__(
      src_name=src_name, src_dir=src_dir, out_dir=out_dir,
      prereqs=['gcc'])
    self.exe_name = os.path.splitext(src_name)[0] + FileNames.EXE_EXT
    self.compile_args = tuple(['gcc',
                               '-o', os.path.join(out_dir, self.exe_name),
                               src_name] + flags)
    self.run_args = tuple([os.path.join(out_dir, self.exe_name)])



class CXXCode(FileBasedCode):

  def __init__(self, src_name, src_dir, out_dir, flags):
    super(CXXCode, self).__init__(
      src_name=src_name, src_dir=src_dir, out_dir=out_dir,
      prereqs=['g++'])
    self.exe_name = os.path.splitext(src_name)[0] + FileNames.EXE_EXT
    self.compile_args = tuple(['g++',
                               '-o', os.path.join(out_dir, self.exe_name),
                               src_name] + flags)
    self.run_args = tuple([os.path.join(out_dir, self.exe_name)])



class JavaCode(FileBasedCode):

  def __init__(self, src_name, src_dir, out_dir,
               encoding, mainclass, compile_flags, run_flags):
    super(JavaCode, self).__init__(
      src_name=src_name, src_dir=src_dir, out_dir=out_dir,
      prereqs=['javac', 'java'])
    self.encoding = encoding
    self.mainclass = mainclass
    self.compile_args = tuple(['javac', '-encoding', encoding,
                               '-d', FileUtil.ConvPath(out_dir)] +
                              compile_flags + [src_name])
    self.run_args = tuple(['java', '-Dline.separator=\n',
                           '-cp', FileUtil.ConvPath(out_dir)] +
                          run_flags + [mainclass])



class ScriptCode(FileBasedCode):

  QUIET_COMPILE = True

  def __init__(self, src_name, src_dir, out_dir, interpreter):
    super(ScriptCode, self).__init__(
      src_name=src_name, src_dir=src_dir, out_dir=out_dir,
      prereqs=[interpreter])
    self.interpreter = interpreter
    self.run_args = tuple([interpreter, os.path.join(out_dir, src_name)])

  @GeneratorTask.FromFunction
  def Compile(self):
    try:
      self.MakeOutDir()
      FileUtil.CopyFile(os.path.join(self.src_dir, self.src_name),
                        os.path.join(self.out_dir, self.src_name))
      result = RunResult(RunResult.OK, 0.0)
    except Exception, e:
      FileUtil.WriteFile(str(e), os.path.join(self.out_dir, self.log_name))
      result = RunResult(RunResult.RE, None)
    yield result


class DiffCode(Code):

  QUIET_COMPILE = True

  def __init__(self):
    super(DiffCode, self).__init__()
    self.log_name = 'diff.log'

  @GeneratorTask.FromFunction
  def Compile(self):
    if not FileUtil.LocateBinary('diff'):
      yield RunResult("%s: diff" % RunResult.PM, None)
    yield RunResult(RunResult.OK, 0.0)

  @GeneratorTask.FromFunction
  def Run(self, args, cwd, input, output, timeout, precise, redirect_error=False):
    parser = optparse.OptionParser()
    parser.add_option('-i', '--infile', dest='infile')
    parser.add_option('-d', '--difffile', dest='difffile')
    parser.add_option('-o', '--outfile', dest='outfile')
    (options, pos_args) = parser.parse_args([''] + list(args))
    run_args = ('diff', '-u', options.difffile, options.outfile)
    with open(input, 'r') as infile:
      with open(output, 'w') as outfile:
        if redirect_error:
          errfile = subprocess.STDOUT
        else:
          errfile = FileUtil.OpenNull()
        task = ExternalProcessTask(
          run_args, cwd=cwd, stdin=infile, stdout=outfile, stderr=errfile,
          timeout=timeout)
        try:
          proc = yield task
        except OSError:
          yield RunResult(RunResult.RE, None)
        ret = proc.returncode
        if ret == 0:
          yield RunResult(RunResult.OK, task.time)
        if ret > 0:
          yield RunResult(RunResult.NG, None)
        yield RunResult(RunResult.RE, None)

  @GeneratorTask.FromFunction
  def Clean(self):
    yield True


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
      except AttributeError:
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

  @GeneratorTask.FromFunction
  def Build(self, ctx):
    """
    Build all.
    """
    results = yield TaskBranch(
      [problem.Build(ctx) for problem in self.problems])
    yield all(results)

  @GeneratorTask.FromFunction
  def Test(self, ctx):
    """
    Test all.
    """
    results = yield TaskBranch(
      [problem.Test(ctx) for problem in self.problems])
    yield list(itertools.chain(*results))

  @GeneratorTask.FromFunction
  def Pack(self, ctx):
    """
    Pack all.
    """
    results = yield TaskBranch(
      [problem.Pack(ctx) for problem in self.problems])
    yield all(results)

  @GeneratorTask.FromFunction
  def Clean(self, ctx):
    """
    Clean all.
    """
    results = yield TaskBranch(
      [problem.Clean(ctx) for problem in self.problems])
    yield all(results)



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

  @GeneratorTask.FromFunction
  def Build(self, ctx):
    """
    Build all solutions and tests.
    """
    results = yield TaskBranch(
      [solution.Build(ctx) for solution in self.solutions] +
      [self.tests.Build(ctx)])
    yield all(results)

  @GeneratorTask.FromFunction
  def Test(self, ctx):
    """
    Run tests.
    """
    yield (yield self.tests.Test(ctx))

  @GeneratorTask.FromFunction
  def Pack(self, ctx):
    """
    Pack tests.
    """
    yield (yield self.tests.Pack(ctx))

  @GeneratorTask.FromFunction
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
    yield success



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

  @GeneratorTask.FromFunction
  def Build(self, ctx):
    """
    Build tests.
    """
    if self.IsBuildCached():
      yield True
    if not self._InitOutputDir(ctx):
      yield False
    if not all((yield TaskBranch([
            self._CompileGenerator(ctx),
            self._CompileValidator(ctx),
            self._CompileJudge(ctx)]))):
      yield False
    if not (yield self._RunGenerator(ctx)):
      yield False
    if not (yield self._RunValidator(ctx)):
      yield False
    if self.ListInputFiles():
      if not (yield self._CompileReferenceSolution(ctx)):
        yield False
      if not (yield self._RunReferenceSolution(ctx)):
        yield False
      if not (yield self._GenerateConcatTest(ctx)):
        yield False
    if not self.SetCacheStamp(ctx):
      yield False
    yield True

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

  @GeneratorTask.FromFunction
  def _CompileGenerator(self, ctx):
    """
    Compile all input generators.
    """
    results = yield TaskBranch([
        self._CompileGeneratorOne(generator, ctx)
        for generator in self.generators])
    yield all(results)

  @GeneratorTask.FromFunction
  def _CompileGeneratorOne(self, generator, ctx):
    """
    Compile a single input generator.
    """
    if not generator.QUIET_COMPILE:
      Console.PrintAction("COMPILE", self, generator.src_name)
    res = yield generator.Compile()
    if res.status != RunResult.OK:
      ctx.errors.Error(self,
                       "%s: Compile Error (%s)" % (generator.src_name, res.status))
      Console.PrintLog(generator.ReadCompileLog())
      raise Bailout([False])
    yield True

  @GeneratorTask.FromFunction
  def _RunGenerator(self, ctx):
    """
    Run all input generators.
    """
    results = yield TaskBranch([
        self._RunGeneratorOne(generator, ctx)
        for generator in self.generators])
    yield all(results)

  @GeneratorTask.FromFunction
  def _RunGeneratorOne(self, generator, ctx):
    """
    Run a single input generator.
    """
    Console.PrintAction("GENERATE", self, generator.src_name)
    res = yield generator.Run(
      args=(), cwd=self.out_dir,
      input=os.devnull, output=os.devnull, timeout=None, precise=False)
    if res.status != RunResult.OK:
      ctx.errors.Error(self,
                       "%s: %s" % (generator.src_name, res.status))
      raise Bailout([False])
    yield True

  @GeneratorTask.FromFunction
  def _CompileValidator(self, ctx):
    """
    Compile input validators.
    """
    results = yield TaskBranch([
        self._CompileValidatorOne(validator, ctx)
        for validator in self.validators])
    yield all(results)

  @GeneratorTask.FromFunction
  def _CompileValidatorOne(self, validator, ctx):
    """
    Compile a single input validator.
    """
    if not validator.QUIET_COMPILE:
      Console.PrintAction("COMPILE", self, validator.src_name)
    res = yield validator.Compile()
    if res.status != RunResult.OK:
      ctx.errors.Error(self,
                       "%s: Compile Error (%s)" % (validator.src_name, res.status))
      Console.PrintLog(validator.ReadCompileLog())
      raise Bailout([False])
    yield True

  @GeneratorTask.FromFunction
  def _RunValidator(self, ctx):
    """
    Run input validators.
    """
    if not self.validators:
      Console.PrintAction("VALIDATE", self, "skipping: validator unavailable")
      ctx.errors.Warning(self, "Validator Unavailable")
      yield True
    infiles = self.ListInputFiles()
    results = yield TaskBranch([
        self._RunValidatorOne(validator, infile, ctx)
        for validator in self.validators
        for infile in infiles])
    if not all(results):
      yield False
    Console.PrintAction("VALIDATE", self, "OK")
    yield True

  @GeneratorTask.FromFunction
  def _RunValidatorOne(self, validator, infile, ctx):
    """
    Run an input validator against a single input file.
    """
    #Console.PrintAction("VALIDATE", self,
    #                    "%s" % infile, progress=True)
    validationfile = os.path.splitext(infile)[0] + FileNames.VALIDATION_EXT
    res = yield validator.Run(
      args=(), cwd=self.out_dir,
      input=os.path.join(self.out_dir, infile),
      output=os.path.join(self.out_dir, validationfile),
      timeout=None, precise=False,
      redirect_error=True)
    if res.status == RunResult.NG:
      ctx.errors.Error(self,
                       "%s: Validation Failed" % infile)
      log = FileUtil.ReadFile(os.path.join(self.out_dir, validationfile))
      Console.PrintLog(log)
      raise Bailout([False])
    elif res.status != RunResult.OK:
      ctx.errors.Error(self,
                       "%s: Validator Failed: %s" % (infile, res.status))
      raise Bailout([False])
    Console.PrintAction("VALIDATE", self,
                        "%s: PASSED" % infile, progress=True)
    yield True

  @GeneratorTask.FromFunction
  def _CompileJudge(self, ctx):
    """
    Compile judge.
    """
    if self.judge is None:
      yield True
    if not self.judge.QUIET_COMPILE:
      Console.PrintAction("COMPILE", self, self.judge.src_name)
    res = yield self.judge.Compile()
    if res.status != RunResult.OK:
      ctx.errors.Error(self, "%s: Compile Error (%s)" % (self.judge.src_name, res.status))
      Console.PrintLog(self.judge.ReadCompileLog())
      yield False
    yield True

  @GeneratorTask.FromFunction
  def _CompileReferenceSolution(self, ctx):
    """
    Compile the reference solution.
    """
    reference_solution = self.problem.reference_solution
    if reference_solution is None:
      ctx.errors.Error(self, "Reference solution is not available")
      yield False
    yield (yield reference_solution.Build(ctx))

  @GeneratorTask.FromFunction
  def _RunReferenceSolution(self, ctx):
    """
    Run the reference solution to generate reference outputs.
    """
    reference_solution = self.problem.reference_solution
    if reference_solution is None:
      ctx.errors.Error(self, "Reference solution is not available")
      yield False
    infiles = self.ListInputFiles()
    results = yield TaskBranch([
        self._RunReferenceSolutionOne(reference_solution, infile, ctx)
        for infile in infiles])
    if not all(results):
      yield False
    Console.PrintAction("REFRUN", reference_solution)
    yield True

  @GeneratorTask.FromFunction
  def _RunReferenceSolutionOne(self, reference_solution, infile, ctx):
    """
    Run the reference solution against a single input file.
    """
    difffile = os.path.splitext(infile)[0] + FileNames.DIFF_EXT
    if os.path.isfile(os.path.join(self.out_dir, difffile)):
      yield True
    #Console.PrintAction("REFRUN", reference_solution,
    #                    "%s" % infile, progress=True)
    res = yield reference_solution.Run(
      args=(), cwd=self.out_dir,
      input=os.path.join(self.out_dir, infile),
      output=os.path.join(self.out_dir, difffile),
      timeout=None, precise=False)
    if res.status != RunResult.OK:
      ctx.errors.Error(reference_solution, res.status)
      raise Bailout([False])
    Console.PrintAction("REFRUN", reference_solution,
                        "%s: DONE" % infile, progress=True)
    yield True

  @GeneratorTask.FromFunction
  def _GenerateConcatTest(self, ctx):
    if not self.concat_test:
      yield True
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
    yield True

  @GeneratorTask.FromFunction
  def Test(self, ctx):
    """
    Test all solutions.
    """
    if not (yield self.Build(ctx)):
      yield []
    results = yield TaskBranch([
        self.TestSolution(solution, ctx)
        for solution in self.problem.solutions])
    yield list(itertools.chain(*results))

  @GeneratorTask.FromFunction
  def TestSolution(self, solution, ctx):
    """
    Test a single solution.
    """
    # Note: though Tests.Test() executes Tests.Build(), it is also
    # required here because Solution.Test() calls this function directly.
    if not (yield self.Build(ctx)):
      result = TestResult(self.problem, solution, [])
      result.good = False
      result.passed = False
      result.detail = "Failed to build tests"
      yield [result]
    if not (yield solution.Build(ctx)):
      result = TestResult(self.problem, solution, [])
      result.good = False
      result.passed = False
      result.detail = "Compile Error"
      yield [result]
    Console.PrintAction("TEST", solution, progress=True)
    if not solution.IsCorrect() and solution.challenge_cases:
      result = yield self._TestSolutionWithChallengeCases(solution, ctx)
    else:
      result = yield self._TestSolutionWithAllCases(solution, ctx)
    if result.good and result.passed:
      assert not result.detail
      if result.IsTimeStatsAvailable(ctx):
        result.detail = result.GetTimeStats()
      else:
        result.detail = "(*/*)"
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
    yield [result]

  @GeneratorTask.FromFunction
  def _TestSolutionWithChallengeCases(self, solution, ctx):
    """
    Test a wrong solution which has explicitly-specified challenge cases.
    """
    infiles = self.ListInputFiles()
    challenge_cases = self._SortInputFiles(solution.challenge_cases)
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
      yield result
    # Try challenge cases.
    yield TaskBranch([
        self._TestSolutionWithChallengeCasesOne(solution, infile, result, ctx)
        for infile in challenge_cases],
        unsafe_interrupt=True)
    if result.good is None:
      result.good = True
      result.passed = False
      result.detail = "Expectedly Failed"
    yield result

  @GeneratorTask.FromFunction
  def _TestSolutionWithChallengeCasesOne(self, solution, infile, result, ctx):
    """
    Test a wrong solution which has explicitly-specified challenge cases.
    """
    #Console.PrintAction("TEST", solution,
    #                    "%s" % infile, progress=True)
    cookie = solution.GetCacheStamp()
    ignore_timeout = (infile == FileNames.CONCAT_INFILE)
    (verdict, time, cached) = yield self._TestOneCase(
      solution, infile, cookie, ignore_timeout, ctx)
    if cached:
      result.cached = True
    result.cases[infile].verdict = verdict
    if verdict == TestResult.AC:
      result.ruling_file = infile
      result.good = False
      result.passed = True
      result.detail = "%s: Unexpectedly Accepted" % infile
      ctx.errors.Error(solution, result.detail)
      if ctx.options.ignore_errors:
        yield False
      else:
        raise Bailout([False])
    elif verdict not in (TestResult.WA, TestResult.TLE, TestResult.RE):
      result.ruling_file = infile
      result.good = False
      result.passed = False
      result.detail = "%s: Judge Error" % infile
      ctx.errors.Error(solution, result.detail)
      if ctx.options.ignore_errors:
        yield False
      else:
        raise Bailout([False])
    Console.PrintAction("TEST", solution,
                        "%s: PASSED" % infile, progress=True)
    yield True

  @GeneratorTask.FromFunction
  def _TestSolutionWithAllCases(self, solution, ctx):
    """
    Test a solution without challenge cases.
    The solution can be marked as wrong but without challenge cases.
    """
    infiles = self.ListInputFiles(include_concat=True)
    result = TestResult(self.problem, solution, infiles)
    # Try all cases.
    yield TaskBranch([
        self._TestSolutionWithAllCasesOne(solution, infile, result, ctx)
        for infile in infiles],
        unsafe_interrupt=True)
    if result.good is None:
      result.good = solution.IsCorrect()
      result.passed = True
      if not result.good:
        result.detail = "Unexpectedly Passed"
    yield result

  @GeneratorTask.FromFunction
  def _TestSolutionWithAllCasesOne(self, solution, infile, result, ctx):
    """
    Test a solution without challenge cases.
    The solution can be marked as wrong but without challenge cases.
    """
    #Console.PrintAction("TEST", solution,
    #                    "%s" % infile, progress=True)
    cookie = solution.GetCacheStamp()
    ignore_timeout = (infile == FileNames.CONCAT_INFILE)
    (verdict, time, cached) = yield self._TestOneCase(
      solution, infile, cookie, ignore_timeout, ctx)
    if cached:
      result.cached = True
    result.cases[infile].verdict = verdict
    if verdict not in (TestResult.AC, TestResult.WA, TestResult.TLE, TestResult.RE):
      result.ruling_file = infile
      result.good = False
      result.passed = False
      result.detail = "%s: Judge Error" % infile
      ctx.errors.Error(solution, result.detail)
      if ctx.options.ignore_errors:
        yield False
      else:
        raise Bailout([False])
    elif verdict != TestResult.AC:
      result.ruling_file = infile
      result.passed = False
      result.detail = "%s: %s" % (infile, verdict)
      if solution.IsCorrect():
        result.good = False
        ctx.errors.Error(solution, result.detail)
      else:
        result.good = True
      if ctx.options.ignore_errors:
        yield False
      else:
        raise Bailout([False])
    result.cases[infile].time = time
    Console.PrintAction("TEST", solution,
                        "%s: PASSED" % infile, progress=True)
    yield True

  @GeneratorTask.FromFunction
  def _TestOneCase(self, solution, infile, cookie, ignore_timeout, ctx):
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
          yield tuple(list(result)+[True])
    result = yield self._TestOneCaseNoCache(solution, infile, ignore_timeout, ctx)
    try:
      FileUtil.PickleSave((cookie, result), cachefile)
    except:
      ctx.errors.Exception(solution)
    yield tuple(list(result)+[False])

  @GeneratorTask.FromFunction
  def _TestOneCaseNoCache(self, solution, infile, ignore_timeout, ctx):
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
    precise = (ctx.options.precise or ctx.options.parallelism == 1)
    res = yield solution.Run(
      args=(), cwd=solution.out_dir,
      input=os.path.join(self.out_dir, infile),
      output=os.path.join(solution.out_dir, outfile),
      timeout=timeout, precise=precise)
    if res.status == RunResult.TLE:
      yield (TestResult.TLE, None)
    if res.status != RunResult.OK:
      yield (TestResult.RE, None)
    time = res.time
    res = yield self.judge.Run(
      args=('--infile', os.path.join(self.out_dir, infile),
            '--difffile', os.path.join(self.out_dir, difffile),
            '--outfile', os.path.join(solution.out_dir, outfile)),
      cwd=self.out_dir,
      input=os.devnull,
      output=os.path.join(solution.out_dir, judgefile),
      timeout=None, precise=False)
    if res.status == RunResult.OK:
      yield (TestResult.AC, time)
    if res.status == RunResult.NG:
      yield (TestResult.WA, None)
    yield ("Validator " + res.status, None)

  @GeneratorTask.FromFunction
  def Pack(self, ctx):
    """
    Pack test cases.
    """
    if self.IsBuildCached():
      # TODO: do caching of packed tests output here.
      pass
    else:
      if not (yield self.Build(ctx)):
        yield False
    infiles = self.ListInputFiles()
    Console.PrintAction("PACK", self, progress=True)
    if not os.path.isdir(self.pack_dir):
      try:
        FileUtil.MakeDir(self.pack_dir)
      except:
        ctx.errors.Exception(self)
        yield False
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
        yield False
    tar_args = ("tar", "czf",
                os.path.join(os.pardir, FileNames.TESTS_PACKED_TARBALL),
                os.curdir)
    Console.PrintAction(
      "PACK",
      self,
      " ".join(tar_args),
      progress=True)
    devnull = FileUtil.OpenNull()
    task = ExternalProcessTask(
      tar_args, cwd=self.pack_dir,
      stdin=devnull, stdout=devnull, stderr=devnull)
    try:
      proc = yield task
    except:
      ctx.errors.Exception(self)
      yield False
    ret = proc.returncode
    if ret != 0:
      ctx.errors.Error(self, "tar failed: ret = %d" % ret)
      yield False
    Console.PrintAction(
      "PACK",
      self,
      FileNames.TESTS_PACKED_TARBALL)
    yield True

  @GeneratorTask.FromFunction
  def Clean(self, ctx):
    """
    Remove test cases.
    """
    Console.PrintAction("CLEAN", self)
    try:
      FileUtil.RemoveTree(self.out_dir)
    except:
      ctx.errors.Exception(self)
    yield True

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

  @GeneratorTask.FromFunction
  def Build(self, ctx):
    """
    Build this solution.
    """
    #Console.PrintAction("BUILD", self)
    if self.IsBuildCached():
      Console.PrintAction("COMPILE", self, "up-to-date")
      yield True
    if not self.code.QUIET_COMPILE:
      Console.PrintAction("COMPILE", self)
    res = yield self.code.Compile()
    log = self.code.ReadCompileLog()
    if res.status != RunResult.OK:
      ctx.errors.Error(self, "Compile Error (%s)" % res.status)
      Console.PrintLog(log)
      yield False
    if log:
      Console.Print("Compiler warnings found:")
      Console.PrintLog(log)
    if not self.SetCacheStamp(ctx):
      yield False
    yield True

  @GeneratorTask.FromFunction
  def Test(self, ctx):
    """
    Test this solution.
    """
    yield (yield self.problem.tests.TestSolution(self, ctx))

  @GeneratorTask.FromFunction
  def Run(self, args, cwd, input, output, timeout, precise):
    """
    Run this solution.
    """
    yield (yield self.code.Run(
        args=args, cwd=cwd, input=input, output=output,
        timeout=timeout, precise=precise))

  @GeneratorTask.FromFunction
  def Clean(self, ctx):
    """
    Clean this solution.
    """
    Console.PrintAction("CLEAN", self)
    e = yield self.code.Clean()
    if e:
      ctx.errors.Exception(self, e)
      yield False
    yield True



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
    # Run the task.
    graph = self.CreateTaskGraph(ctx)
    try:
      if cmd == 'build':
        success = graph.Run(obj.Build(ctx))
        Console.Print("Finished Build.")
      elif cmd == 'test':
        results = graph.Run(obj.Test(ctx))
        Console.Print("Finished Test.")
        Console.Print()
        self.PrintTestSummary(results, ctx)
      elif cmd == 'clean':
        success = graph.Run(obj.Clean(ctx))
        Console.Print("Finished Clean.")
      elif cmd == 'pack':
        success = graph.Run(obj.Pack(ctx))
        Console.Print("Finished Pack.")
      else:
        Console.PrintError("Unknown command: %s" % cmd)
        return 1
    except KeyboardInterrupt:
      if ctx.options.debug >= 1:
        traceback.print_exc()
      raise
    finally:
      graph.Close()
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

  def CreateTaskGraph(self, ctx):
    """
    Create the instance of TaskGraph to use for this session.
    """
    if ctx.options.parallelism == 1:
      graph = SerialTaskGraph()
    else:
      graph = FiberTaskGraph(parallelism=ctx.options.parallelism,
                             debug=ctx.options.debug)
    return graph

  def PrintHelp(self):
    """
    Just print help message.
    """
    print HELP_MESSAGE

  def PrintTestSummary(self, results, ctx):
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
          if result.IsTimeStatsAvailable(ctx):
            status_row += [result.GetTimeStats()]
          else:
            status_row += ["(*/*)"]
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
    if not (ctx.options.precise or ctx.options.parallelism == 1):
      Console.Print()
      Console.Print("Note: Timings are not displayed when "
                    "parallel testing is enabled.")
      Console.Print("      To show them, try -p (--precise).")

  def GetOptionParser(self):
    """
    Construct optparse.OptionParser object for Rime.
    """
    parser = optparse.OptionParser(add_help_option=False)
    parser.add_option('-j', '--jobs', dest='parallelism',
                      default=1, action="store", type="int")
    parser.add_option('-p', '--precise', dest='precise',
                      default=False, action="store_true")
    parser.add_option('-i', '--ignore-errors', dest='ignore_errors',
                      default=False, action="store_true")
    parser.add_option('-C', '--cache-tests', dest='cache_tests',
                      default=False, action="store_true")
    parser.add_option('-d', '--debug', dest='debug',
                      default=0, action="count")
    parser.add_option('-h', '--help', dest='show_help',
                      default=False, action="store_true")
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
