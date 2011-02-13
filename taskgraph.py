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


# State of tasks.
RUNNING, WAITING, BLOCKED, READY, FINISHED, ABORTED = range(6)


class TaskBranch(object):

  def __init__(self, tasks):
    self.tasks = tasks


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

  def Wait(self):
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

  def Wait(self):
    assert self.proc is not None
    self.proc.wait()

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


class FiberTaskGraph(object):
  """
  TaskGraph which executes tasks with fibers (microthreads).

  FiberTaskGraph allows some tasks to be in blocked state in the same time.
  Branched tasks are executed in arbitrary order.
  """

  def __init__(self, parallelism):
    self.parallelism = parallelism
    self.cache = dict()
    self.task_graph = dict()
    self.task_counters = dict()
    self.task_waits = dict()
    self.task_state = dict()
    self.ready_tasks = []
    self.blocked_tasks = []
    self.pending_stack = []

  def Close(self):
    pass

  def Run(self, init_task):
    self._BranchTask(None, [init_task])
    while self._RunNextTask():
      pass
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
    assert self.task_state[next_task] == READY
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
    logging.debug('_ContinueTask: %s: entering' % task)
    try:
      result = task.Continue(value)
    except:
      result = _TaskRaise(*sys.exc_info())
    logging.debug('_ContinueTask: %s: exited' % task)
    self._ProcessTaskResult(task, result)

  def _ThrowTask(self, task, exc_info):
    assert self.task_state[task] == RUNNING
    assert not task.IsExclusive() or len(self.blocked_tasks) == 0
    logging.debug('_ThrowTask: %s: entering' % task)
    try:
      result = task.Throw(*exc_info)
    except:
      result = _TaskRaise(*sys.exc_info())
    logging.debug('_ThrowTask: %s: exited' % task)
    self._ProcessTaskResult(task, result)

  def _ProcessTaskResult(self, task, result):
    assert self.task_state[task] == RUNNING
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
    elif isinstance(result, _TaskRaise):
      logging.debug('_ProcessTaskResult: %s: received exception' % task)
      self._ExceptTask(task, result.exc_info)
    else:
      logging.debug('_ProcessTaskResult: %s: received unknown type,'
                    'implying TaskReturn' % task)
      self._FinishTask(task, result)

  def _BranchTask(self, task, subtasks):
    assert task is None or self.task_state[task] == RUNNING
    self.task_graph[task] = subtasks
    if not isinstance(subtasks, list):
      assert isinstance(subtasks, Task)
      subtasks = [subtasks]
    if len(subtasks) == 0:
      logging.debug('_BranchTask: %s: zero branch, fast return' % task)
      self.ready_tasks.insert(0, task)
      self._SetTaskState(task, READY)
      self._LogTaskStats()
      return
    self.task_counters[task] = len(subtasks)
    # The branches are half-expanded, but don't complete the operation here
    # so that too many branches are opened.
    for subtask in reversed(subtasks):
      self.pending_stack.append((task, subtask))
    self._SetTaskState(task, WAITING)

  def _BeginTask(self, task, parent_task):
    if task in self.cache:
      assert self.task_state[task] in (FINISHED, ABORTED)
      logging.debug('_BeginTask: %s: cache hit' % task)
      success = self.cache[task][0]
      if success:
        self._ResolveTask(parent_task)
      else:
        self._BailoutTask(parent_task)
    elif parent_task not in self.task_counters:
      # Some sibling task already bailed out. Skip this task.
      logging.debug('_BeginTask: %s: sibling task bailed out' % task)
      return
    else:
      if task in self.task_waits:
        assert self.task_state[task] in (WAITING, BLOCKED)
        logging.debug('_BeginTask: %s: running' % task)
        self.task_waits[task].append(parent_task)
      else:
        assert task not in self.task_state
        logging.debug('_BeginTask: %s: starting' % task)
        self.task_waits[task] = [parent_task]
        self._SetTaskState(task, RUNNING)
        if task.IsExclusive():
          self._WaitBlockedTasksUntilEmpty()
        self._ContinueTask(task, None)

  def _FinishTask(self, task, value):
    assert self.task_state[task] == RUNNING
    try:
      task.Close()
    except RuntimeError:
      # Python2.5 raises RuntimeError when GeneratorExit is ignored. This often
      # happens when yielding a return value from inside of try block, or even
      # Ctrl+C was pressed when in try block.
      pass
    except:
      self._ExceptTask(task, sys.exc_info())
      return
    self.cache[task] = (True, value)
    logging.debug('_FinishTask: %s: finished, returned: %s' % (task, value))
    for wait_task in self.task_waits[task]:
      self._ResolveTask(wait_task)
    del self.task_waits[task]
    self._SetTaskState(task, FINISHED)

  def _ExceptTask(self, task, exc_info):
    assert self.task_state[task] in (RUNNING, BLOCKED)
    assert task not in self.cache
    self.cache[task] = (False, exc_info)
    logging.debug('_ExceptTask: %s: exception raised: %s' %
                  (task, exc_info[0].__name__))
    for wait_task in self.task_waits[task]:
      self._BailoutTask(wait_task)
    del self.task_waits[task]
    if self.task_state[task] == BLOCKED:
      del self.task_counters[task]
    self._SetTaskState(task, ABORTED)

  def _BlockTask(self, task):
    assert self.task_state[task] == RUNNING
    assert len(self.blocked_tasks) < self.parallelism
    self.task_counters[task] = 1
    self.blocked_tasks.insert(0, task)
    self._SetTaskState(task, BLOCKED)
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
      assert self.task_state[task] == BLOCKED
      try:
        success = task.Poll()
      except:
        self._ExceptTask(task, sys.exc_info())
        resolved += 1
        self.blocked_tasks.pop(i)
        self._LogTaskStats()
      else:
        if success:
          self._ResolveTask(task)
          resolved += 1
          self.blocked_tasks.pop(i)
          self._LogTaskStats()
        else:
          i += 1
    return resolved

  def _ResolveTask(self, task):
    assert task is None or self.task_state[task] in (WAITING, BLOCKED), "%s:%d" % (task, self.task_state[task])
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
      self._SetTaskState(task, READY)
      logging.debug('_ResolveTask: %s: pushed to ready_task' % task)
      self._LogTaskStats()

  def _BailoutTask(self, task):
    if task not in self.task_counters:
      logging.debug('_BailoutTask: %s: multiple bail out' % task)
      return
    assert self.task_state[task] in (WAITING, BLOCKED)
    logging.debug('_BailoutTask: %s: bailing out' % task)
    if task in self.task_graph and isinstance(self.task_graph[task], list):
      # Multiple branches.
      self.ready_tasks.append(task)
    else:
      # Serial execution or blocked task.
      self.ready_tasks.insert(0, task)
    del self.task_counters[task]
    self._SetTaskState(task, READY)
    logging.debug('_BailoutTask: %s: pushed to ready_task' % task)

  def _Sleep(self):
    # TODO(nya): Get rid of this.
    time.sleep(0.01)

  def _LogTaskStats(self):
    logging.info('Task statistics: %d ready, %d blocked, %d opened, %d pending' %
                 (len(self.ready_tasks), len(self.blocked_tasks),
                  len(self.task_waits), len(self.task_counters)))

  def _SetTaskState(self, task, state):
    if state == RUNNING:
      assert task not in self.cache
      assert task not in self.task_graph
      assert task not in self.task_counters
      assert task is None or task in self.task_waits
    elif state == WAITING:
      assert task not in self.cache
      assert task in self.task_graph
      assert task in self.task_counters
      assert task is None or task in self.task_waits
    elif state == BLOCKED:
      assert task not in self.cache
      assert task not in self.task_graph
      assert self.task_counters.get(task) == 1
      assert task in self.task_waits
    elif state == READY:
      assert task not in self.cache
      assert task not in self.task_counters
      assert task is None or task in self.task_waits
    elif state == FINISHED:
      assert task in self.cache and self.cache[task][0]
      assert task not in self.task_graph
      assert task not in self.task_counters
      assert task not in self.task_waits
    elif state == ABORTED:
      assert task in self.cache and not self.cache[task][0]
      assert task not in self.task_graph
      assert task not in self.task_counters
      assert task not in self.task_waits
    else:
      raise AssertionError('Unknown state: ' + str(state))
    self.task_state[task] = state


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
            result = task.Continue(value[1].value)
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
          try:
            task.Wait()
          except:
            self.cache[task] = (False, sys.exc_info())
            break
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
      except RuntimeError:
        pass
      except:
        self.cache[task] = (False, sys.exc_info())
    if self.cache[task] is None:
      raise RuntimeException('Cyclic task dependency found')
    success, value = self.cache[task]
    if success:
      return value
    elif isinstance(value[1], Bailout):
      return value[1].value
    else:
      raise value[0], value[1], value[2]



######## Experimental: multi-threaded blocked task control


class ThreadPool(object):

  def __init__(self, size):
    import Queue
    self.size = size
    self.jobs = Queue.Queue(size)
    self.results = Queue.Queue()
    self.threads = []
    for i in range(size):
      th = threading.Thread(target=self._Worker)
      th.start()
      self.threads.append(th)

  def Close(self):
    for i in range(self.size):
      self.jobs.put((self._Exit, (), {}))
    for th in self.threads:
      th.join()
    self.threads = None
    self.jobs = None
    self.results = None

  def Put(self, func, *args, **kwargs):
    self.jobs.put((func, args, kwargs))

  def Get(self):
    return self.results.get()

  def _Worker(self):
    while True:
      func, args, kwargs = self.jobs.get()
      try:
        result = func(*args, **kwargs)
      except SystemExit:
        break
      self.results.put(result)

  def _Exit(self):
    raise SystemExit()


class FiberMTTaskGraph(FiberTaskGraph):

  def __init__(self, parallelism):
    super(FiberMTTaskGraph, self).__init__(parallelism)
    self.thread_pool = ThreadPool(parallelism)

  def Close(self):
    self.thread_pool.Close()

  def _BlockTask(self, task):
    assert self.task_state[task] == RUNNING
    assert len(self.blocked_tasks) < self.parallelism
    self.task_counters[task] = 1
    self.blocked_tasks.insert(0, task)
    self._SetTaskState(task, BLOCKED)
    self.thread_pool.Put(self._WatchBlockedTask, task)
    self._LogTaskStats()
    logging.debug('_BlockTask: %s: pushed to blocked_tasks' % task)
    self._WaitBlockedTasksUntilNotFull()
    assert len(self.blocked_tasks) < self.parallelism

  def _WaitBlockedTasks(self):
    assert len(self.blocked_tasks) > 0
    self._LogTaskStats()
    logging.debug('_WaitBlockedTasks: waiting')
    task, exc_info = self.thread_pool.Get()
    if exc_info is None:
      self._ResolveTask(task)
    else:
      self._ExceptTask(task, exc_info)
    self.blocked_tasks.remove(task)
    self._LogTaskStats()
    return 1

  def _WatchBlockedTask(self, task):
    try:
      task.Wait()
    except:
      return (task, sys.exc_info())
    return (task, None)



######## Samples


class Sample1(object):

  def Main(self):
    graph = FiberTaskGraph(parallelism=5)
    return graph.Run(self.Run())

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
    graph = FiberTaskGraph(parallelism=5)
    return graph.Run(self.Find('/etc'))

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
    graph = FiberTaskGraph(parallelism=5)
    return graph.Run(self.Run())

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
