#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2011 Shuhei Takahashi
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

"""A framework for parallel processing in single-threaded environment."""

import functools
import inspect
import logging
import os
import signal
import subprocess
import sys
import threading
import time


class Task(object):

  def __hash__(self):
    if self.CacheKey() is None:
      return id(self)
    return hash(self.CacheKey())

  def __eq__(self, other):
    if not isinstance(other, Task):
      return False
    if self.CacheKey() is None and other.CacheKey() is None:
      return id(self) == id(other)
    return self.CacheKey() == other.CacheKey()

  def IsExclusive(self):
    return False

  def IsCacheable(self):
    return self.CacheKey() is not None

  def CacheKey(self):
    raise NotImplementedError()

  def Continue(self, value=None):
    raise NotImplementedError()

  def Throw(self, exception):
    raise NotImplementedError()

  def Poll(self):
    raise NotImplementedError()

  def Close(self):
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
    self.it.close()

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

  def _StartProcess(self):
    self.start_time = time.time()
    self.proc = subprocess.Popen(*self.args, **self.kwargs)
    if self.timeout is not None:
      def TimeoutKiller():
        try:
          os.kill(self.proc.pid, signal.SIGKILL)
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


class TaskPool(object):

  def __init__(self, parallelism):
    self.parallelism = parallelism
    self.cache = dict()
    self.task_graph = dict()
    self.task_counters = dict()
    self.task_waits = dict()
    self.ready_tasks = []
    self.blocked_tasks = []
    self.branch_stack = []

  def Run(self, init_task):
    self._BranchTask(None, [init_task])
    while self._RunNextTask():
      pass
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
      if not self._ExpandBranch():
        self._WaitBlockedTasks()
    next_task = self.ready_tasks.pop(0)
    self._LogTaskStats()
    if next_task is None:
      return False
    assert next_task not in self.task_counters
    exc_info = None
    if next_task in self.task_graph:
      if isinstance(self.task_graph[next_task], list):
        value = []
        for task in self.task_graph[next_task]:
          if task in self.cache:
            success, cached = self.cache[task]
            if success:
              value.append(cached)
            else:
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
    if exc_info is not None:
      if isinstance(exc_info[1], Bailout):
        self._ContinueTask(next_task, exc_info[1].value)
      else:
        self._ThrowTask(next_task, exc_info)
    else:
      self._ContinueTask(next_task, value)
    return True

  def _ExpandBranch(self):
    if not self.branch_stack:
      return False
    # Visit branches by depth first.
    task, subtask = self.branch_stack.pop()
    self._BeginTask(subtask, task)
    return True

  def _ThrowTask(self, task, exc_info):
    assert not task.IsExclusive() or len(self.blocked_tasks) == 0
    assert task not in self.task_graph
    assert task not in self.task_counters
    assert task not in self.cache
    try:
      logging.debug('_ThrowTask: %s: entering' % task)
      result = task.Throw(*exc_info)
      logging.debug('_ThrowTask: %s: exited' % task)
      self._ProcessTaskResult(task, result)
    except:
      logging.debug('_ThrowTask: %s: exception raised' % task)
      self._ProcessTaskException(task, sys.exc_info())

  def _ContinueTask(self, task, value):
    assert not task.IsExclusive() or len(self.blocked_tasks) == 0
    assert task not in self.task_graph
    assert task not in self.task_counters
    assert task not in self.cache
    try:
      logging.debug('_ContinueTask: %s: entering' % task)
      result = task.Continue(value)
      logging.debug('_ContinueTask: %s: exited' % task)
      self._ProcessTaskResult(task, result)
    except:
      logging.debug('_ContinueTask: %s: exception raised' % task)
      self._ProcessTaskException(task, sys.exc_info())

  def _ProcessTaskResult(self, task, result):
    if isinstance(result, Task):
      logging.debug('_ProcessTaskResult: %s: received Task' % task)
      self._BranchTask(task, result)
    elif isinstance(result, TaskBranch):
      logging.debug('_ProcessTaskResult: %s: received TaskBranch '
                    'with %d tasks' % (task, len(result.tasks)))
      self._BranchTask(task, list(result.tasks))
    elif isinstance(result, TaskReturn):
      logging.debug('_ProcessTaskResult: %s: received TaskReturn' % task)
      self._FinishTask(task, result.value)
    elif isinstance(result, TaskBlock):
      logging.debug('_ProcessTaskResult: %s: received TaskBlock' % task)
      self._BlockTask(task)
    else:
      logging.debug('_ProcessTaskResult: %s: received unknown type,'
                    'implying TaskReturn' % task)
      self._FinishTask(task, result)

  def _BranchTask(self, task, subtasks):
    assert task not in self.task_graph
    assert task not in self.task_counters
    assert task not in self.cache
    self.task_graph[task] = subtasks
    if not isinstance(subtasks, list):
      assert isinstance(subtasks, Task)
      subtasks = [subtasks]
    if len(subtasks) == 0:
      logging.debug('_BranchTask: %s: zero branch, fast return' % task)
      self.ready_tasks.insert(0, task)
      self._LogTaskStats()
      return
    self.task_counters[task] = len(subtasks)
    # The branches are half-expanded, but don't complete the operation here
    # so that too many branches are opened.
    for subtask in reversed(subtasks):
      self.branch_stack.append((task, subtask))

  def _BeginTask(self, task, parent_task):
    if task in self.cache:
      logging.debug('_BeginTask: %s: cache hit' % task)
      success = self.cache[task][0]
      if success:
        self._ResolveTask(parent_task)
      else:
        self._BailoutTask(parent_task)
    elif parent_task not in self.task_counters:
      # Some sibling task already bailed out.
      logging.debug('_BeginTask: %s: sibling task bailed out' % task)
      return
    else:
      if task in self.task_waits:
        logging.debug('_BeginTask: %s: running' % task)
        self.task_waits[task].append(parent_task)
      else:
        logging.debug('_BeginTask: %s: starting' % task)
        self.task_waits[task] = [parent_task]
        if task.IsExclusive():
          self._WaitBlockedTasksUntilEmpty()
        self._ContinueTask(task, None)

  def _FinishTask(self, task, value):
    assert task not in self.cache
    try:
      task.Close()
    except RuntimeError:
      # Python2.5 raises RuntimeError when GeneratorExit is ignored. This often
      # happens when yielding a return value from inside of try block, or even
      # Ctrl+C was pressed when in try block.
      pass
    except:
      self._ProcessTaskException(task, sys.exc_info())
      return
    self.cache[task] = (True, value)
    logging.debug('_FinishTask: %s: finished, returned: %s' % (task, value))
    if task in self.task_waits:
      for wait_task in self.task_waits[task]:
        self._ResolveTask(wait_task)
      del self.task_waits[task]

  def _ProcessTaskException(self, task, exc_info):
    assert task not in self.cache
    self.cache[task] = (False, exc_info)
    logging.debug('_FinishTask: %s: exception raised: %s' %
                  (task, exc_info[0].__name__))
    if task in self.task_waits:
      for wait_task in self.task_waits[task]:
        self._BailoutTask(wait_task)
      del self.task_waits[task]

  def _BlockTask(self, task):
    assert len(self.blocked_tasks) < self.parallelism
    self.task_counters[task] = 1
    self.blocked_tasks.insert(0, task)
    self._LogTaskStats()
    logging.debug('_BlockTask: %s: pushed to blocked_tasks' % task)
    self._WaitBlockedTasksUntilNotFull()
    assert len(self.blocked_tasks) < self.parallelism

  def _WaitBlockedTasksUntilEmpty(self):
    logging.debug('_WaitBlockedTasksUntilEmpty: %d blocked tasks' %
                  len(self.blocked_tasks))
    while len(self.blocked_tasks) > 0:
      self._WaitBlockedTasks()

  def _WaitBlockedTasksUntilNotFull(self):
    logging.debug('_WaitBlockedTasksUntilNotFull: %d blocked tasks' %
                  len(self.blocked_tasks))
    if len(self.blocked_tasks) == self.parallelism:
      logging.info('Maximum parallelism reached, waiting for blocked tasks')
      self._WaitBlockedTasks()

  def _WaitBlockedTasks(self):
    assert len(self.blocked_tasks) > 0
    self._LogTaskStats()
    logging.debug('_WaitBlockedTasks: waiting')
    while True:
      resolved = self._PollBlockedTasks()
      if resolved > 0:
        break
      self._Sleep()
    logging.debug('_WaitBlockedTasks: resolved %d blocked tasks' % resolved)

  def _PollBlockedTasks(self):
    i = 0
    resolved = 0
    while i < len(self.blocked_tasks):
      task = self.blocked_tasks[i]
      if task.Poll():
        self._ResolveTask(task)
        resolved += 1
        self.blocked_tasks.pop(i)
        self._LogTaskStats()
      else:
        i += 1
    return resolved

  def _ResolveTask(self, task):
    if task not in self.task_counters:
      logging.debug('_ResolveTask: %s: resolved, but already bailed out' % task)
      return
    logging.debug('_ResolveTask: %s: resolved, counter: %d -> %d' %
                  (task, self.task_counters[task], self.task_counters[task]-1))
    self.task_counters[task] -= 1
    if self.task_counters[task] == 0:
      if task in self.task_graph and isinstance(self.task_graph[task], list):
        # Multiple branches.
        self.ready_tasks.append(task)
      else:
        # Serial execution or blocked task.
        self.ready_tasks.insert(0, task)
      del self.task_counters[task]
      logging.debug('_ResolveTask: %s: pushed to ready_task' % task)
      self._LogTaskStats()

  def _BailoutTask(self, task):
    if task not in self.task_counters:
      logging.debug('_BailoutTask: %s: multiple bail out' % task)
      return
    logging.debug('_BailoutTask: %s: bailing out' % task)
    if task in self.task_graph and isinstance(self.task_graph[task], list):
      # Multiple branches.
      self.ready_tasks.append(task)
    else:
      # Serial execution or blocked task.
      self.ready_tasks.insert(0, task)
    del self.task_counters[task]
    logging.debug('_BailoutTask: %s: pushed to ready_task' % task)

  def _Sleep(self):
    # TODO(nya): Get rid of this.
    time.sleep(0.01)

  def _LogTaskStats(self):
    logging.info('Task statistics: %d ready, %d blocked, %d opened, %d pending' %
                 (len(self.ready_tasks), len(self.blocked_tasks),
                  len(self.task_waits), len(self.task_counters)))


class TaskBranch(object):

  def __init__(self, tasks):
    self.tasks = tasks


class TaskReturn(object):

  def __init__(self, value):
    self.value = value


class TaskBlock(object):

  pass


class Bailout(Exception):

  def __init__(self, value=None):
    self.value = value



######## Samples


class Sample1(object):

  def Main(self):
    return Task.Boot(self.Run(), parallelism=5)

  @GeneratorTask.FromFunction
  def Run(self):
    results = yield TaskBranch([self.Job(i) for i in range(10)])
    results = self.Concat(results)
    yield TaskReturn(results)

  @GeneratorTask.FromFunction
  def Job(self, i):
    results = yield TaskBranch([self.SubJob(min(i, j), max(i, j)) for j in range(10)])
    yield TaskReturn(results)

  @GeneratorTask.FromFunction
  def SubJob(self, i, j):
    yield TaskReturn(i*j)

  def Concat(self, items):
    concated = []
    for item in items:
      concated.extend(item)
    return concated


class FindTask(ExternalProcessTask):

  def __init__(self, path):
    super(FindTask, self).__init__(
      ['find', path, '-mindepth', '1', '-maxdepth', '1', '-type', 'd'],
      stdout=subprocess.PIPE)
    self.output = ''

  def Poll(self):
    self.output += self.proc.stdout.read(4096)
    return super(FindTask, self).Poll()


class Sample2(object):

  def Main(self):
    return Task.Boot(self.Find('/etc'), parallelism=5)

  @GeneratorTask.FromFunction
  def Find(self, path):
    print "Find:", path
    find = FindTask(path)
    yield TaskBranch([find])
    result = [path]
    output = find.output.strip()
    if output:
      subpaths = output.split('\n')
      subresults = yield TaskBranch([self.Find(subpath) for subpath in subpaths])
      for subresult in subresults:
        result.extend(subresult)
    yield TaskReturn(result)


class Sample3(object):

  def Main(self):
    return Task.Boot(self.Run(), parallelism=5)

  @GeneratorTask.FromFunction
  def Run(self):
    import random
    yield TaskBranch([self.Wait(i, random.randint(1, 5))
                      for i in range(10)])
    print "finished."

  @GeneratorTask.FromFunction
  def Wait(self, i, t):
    print 'Wait(%d, %d): start' % (i, t)
    yield ExternalProcessTask(['/bin/sleep', str(t)])
    print 'Wait(%d, %d): end' % (i, t)


def main():
  #logging.basicConfig(level=logging.DEBUG)
  sample = Sample3()
  print sample.Main()


if __name__ == '__main__':
  main()
