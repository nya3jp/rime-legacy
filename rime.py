#!/usr/bin/python
## -*- coding: utf-8; mode: python -*-
##
## Copyright (c) 2010 TAKAHASHI, Shuhei
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
## THE SOFTWARE.
##

import sys
import os
import time
import re
from optparse import OptionParser
import imp
import subprocess
import threading
import signal
import shutil
import pickle


class FileNames(object):

    RIMEROOT_FILE = 'RIMEROOT'
    PROBLEM_FILE = 'PROBLEM'
    SOLUTION_FILE = 'SOLUTION'
    TESTS_FILE = 'TESTS'

    STAMP_FILE = '.stamp'

    RIME_OUT_DIR = 'rime-out'
    TESTS_DIR = 'tests'

    IN_EXT = '.in'
    DIFF_EXT = '.diff'
    OUT_EXT = '.out'
    EXE_EXT = '.exe'
    JUDGE_EXT = '.judge'
    CACHE_EXT = '.cache'


class RimeOptions(object):

    cache_tests = True
    use_indicator = False


class FileUtil(object):

    @classmethod
    def CopyFile(cls, src, dst):
        try:
            shutil.copy(src, dst)
            return True
        except:
            return False

    @classmethod
    def MakeDir(cls, dir):
        try:
            if not os.path.isdir(dir):
                os.makedirs(dir)
            return True
        except:
            return False

    @classmethod
    def CopyTree(cls, src, dst):
        try:
            shutil.copytree(src, dst)
            return True
        except:
            return False

    @classmethod
    def RemoveTree(cls, dir):
        try:
            if os.path.exists(dir):
                shutil.rmtree(dir)
            return True
        except:
            return False

    @classmethod
    def GetModified(cls, file):
        return os.path.getmtime(file)

    @classmethod
    def Touch(cls, file):
        try:
            f = open(file, 'a')
            f.close()
            return True
        except:
            return False

    @classmethod
    def PickleSave(cls, obj, file):
        try:
            f = open(file, 'w')
            pickle.dump(obj, f)
            return True
        except:
            try:
                f.close()
            except:
                pass
            return False

    @classmethod
    def PickleLoad(cls, file):
        try:
            f = open(file, 'r')
            obj = pickle.load(f)
            return obj
        except:
            try:
                f.close()
            except:
                pass
            return None



class Console(object):

    @classmethod
    def Print(cls, msg, overwrite=False):
        if overwrite:
            print "\x1b[1A%s\x1b[K" % msg
        else:
            print msg

    @classmethod
    def PrintAction(cls, action, obj, msg=None, overwrite=False):
        head = "\x1b[32m" + ("[" + action + "]").ljust(11, ' ') + "\x1b[0m "
        if not msg:
            cls.Print("%s%s" % (head, obj.fullname), overwrite)
        else:
            cls.Print("%s%s: %s" % (head, obj.fullname, msg), overwrite)

    @classmethod
    def PrintError(cls, msg):
        cls.Print("\x1b[31mERROR:\x1b[0m %s" % msg)

    @classmethod
    def PrintWarning(cls, msg):
        cls.Print("\x1b[33mWARNING:\x1b[0m %s" % msg)

    @classmethod
    def PrintLog(cls, log):
        print log,

    @classmethod
    def PrintTitle(cls, title):
        cls.Print("")
        cls.Print("\x1b[1m%s:\x1b[0m" % title)


class TestResult(object):

    DEF = "."
    RUN = "\x1b[1m#\x1b[0m"
    AC = "\x1b[36m*\x1b[0m"
    WA = "\x1b[31mW\x1b[0m"
    TLE = "\x1b[31mT\x1b[0m"
    RE = "\x1b[31mE\x1b[0m"
    ERR = "\x1b[1;41m!\x1b[0m"

    PASSED = "\x1b[1mPASSED\x1b[0m"
    FAILED = "\x1b[1;31mFAILED\x1b[0m"

    def __init__(self, problem, solution, files):
        self.problem = problem
        self.solution = solution
        self.files = files[:]
        self.status = dict()
        self.times = dict()
        for file in self.files:
            self.status[file] = self.DEF
            self.times[file] = None
        self.result = None
        self.detail = None
        self.ruling_file = None

    def SetStatus(self, file, status):
        self.status[file] = status

    def SetTime(self, file, time):
        self.times[file] = time

    def GetIndicator(self):
        s = ""
        for file in self.files:
            s += self.status[file]
        return s

    def IsAllAccepted(self):
        return all([x == self.AC for x in self.status.values()])

    def GetAnyAccepted(self):
        for file in self.files:
            if self.status[file] == self.AC:
                return file
        return None

    def GetAnyRejected(self):
        for file in self.files:
            if self.status[file] != self.DEF and self.status[file] != self.AC:
                return file
        return None

    def GetRejectReason(self, file):
        status = self.status[file]
        if status == self.WA:
            return "Wrong Answer"
        if status == self.TLE:
            return "Time Limit Exceeded"
        if status == self.RE:
            return "Runtime Error"
        if status == self.ERR:
            return "System Error"
        raise ValueError()

    def HasTimeStats(self):
        return (self.files and
                all([x is not None for x in self.times.values()]))

    def GetTimeStats(self):
        max_time = max(self.times.values())
        total_time = sum(self.times.values())
        return "max: %.2f, total: %.2f" % (max_time, total_time)

    @classmethod
    def Compare(self, a, b):
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



class ActionResults(object):

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.test_results = []

    def Error(self, source, reason, quiet=False):
        if source:
            self.errors.append("%s: %s" % (source.fullname, reason))
        else:
            self.errors.append(reason)
        if not quiet:
            Console.PrintError(reason)

    def Warning(self, source, reason, quiet=False):
        if source:
            self.warnings.append("%s: %s" % (source.fullname, reason))
        else:
            self.warnings.append(reason)
        if not quiet:
            Console.PrintWarning(reason)

    def HasError(self):
        return (len(self.errors) > 0)

    def AddTestResult(self, test_result):
        self.test_results.append(test_result)

    def PrintTestSummary(self):
        if len(self.test_results) == 0:
            return
        problem_name_width = max(
            map(lambda t: len(t.problem.name), self.test_results))
        solution_name_width = max(
            map(lambda t: len(t.solution.name), self.test_results))
        last_problem = None
        for test_result in sorted(self.test_results, TestResult.Compare):
            line = ""
            if last_problem is not test_result.problem:
                line += "\x1b[36m"
                line += test_result.problem.name.ljust(problem_name_width)
                line += "\x1b[0m"
                line += " "
                line += ("%d solutions, %d tests" %
                         (len(test_result.problem.solutions),
                          len(test_result.problem.tests._ListInputFiles())))
                Console.Print(line)
                line = ""
                last_problem = test_result.problem
            line += " " * problem_name_width
            line += " "
            if test_result.solution.IsCorrect():
                line += "\x1b[32m"
            else:
                line += "\x1b[33m"
            line += test_result.solution.name.ljust(solution_name_width)
            line += "\x1b[0m "
            line += test_result.result
            line += " "
            if RimeOptions.use_indicator:
                line += test_result.GetIndicator()
                if test_result.detail:
                    line += "\x1b[31m%s\x1b[0m" % test_result.detail
                if test_result.ruling_file:
                    line += " [%s]" % test_result.ruling_file
                if test_result.HasTimeStats():
                    line += " (%s)" % test_result.GetTimeStats()
            else:
                if test_result.result == TestResult.PASSED:
                    if test_result.HasTimeStats():
                        line += "(%s)" % test_result.GetTimeStats()
                else:
                    if test_result.detail:
                        line += "\x1b[31m%s\x1b[0m" % test_result.detail
                    else:
                        if test_result.ruling_file:
                            line += ("\x1b[31m%s\x1b[0m: \x1b[1m%s\x1b[0m" %
                                     (test_result.GetRejectReason(
                                          test_result.ruling_file),
                                      test_result.ruling_file))
                        else:
                            line += "\x1b[31mUnexpectedly Accepted\x1b[0m"
            Console.Print(line)

    def PrintErrorSummary(self):
        for e in self.errors:
            Console.PrintError(e)
        for e in self.warnings:
            Console.PrintWarning(e)
        Console.Print(("Total %d errors, %d warnings" %
                       (len(self.errors), len(self.warnings))))


all_results = ActionResults()



class RunResult(object):

    OK = 'OK'
    NG = 'NG'
    RE = 'RE'
    TLE = 'TLE'

    def __init__(self, status, time):
        self.status = status
        self.time = time


class Code(object):

    COMPILE_LOG_FILE = 'compile.log'
    QUIET_COMPILE = False

    src_name = None
    src_dir = None
    out_dir = None

    def __init__(self, src_name, src_dir, out_dir):
        self.src_name = src_name
        self.src_dir = src_dir
        self.out_dir = out_dir

    def MakeOut_Dir(self):
        FileUtil.MakeDir(self.out_dir)

    def Compile(self):
        self.MakeOut_Dir()
        result = self._ExecForCompile(args=self.compile_args)
        log = self._ReadCompileLog()
        return (result, log)

    def Run(self, args, cwd, input, output, timeout):
        return self._ExecForRun(
            args=self.run_args+args, cwd=cwd,
            input=input, output=output, timeout=timeout)

    def Clean(self):
        return FileUtil.RemoveTree(self.out_dir)

    def _ExecForCompile(self, args):
        try:
            devnull = open(os.devnull, 'w')
            outfile = open(os.path.join(self.out_dir, self.COMPILE_LOG_FILE), 'w')
            return self._ExecInternal(
                args=args, cwd=self.src_dir,
                stdin=devnull, stdout=outfile, stderr=subprocess.STDOUT)
        finally:
            try:
                devnull.close()
                logfile.close()
            except:
                pass

    def _ReadCompileLog(self):
        try:
            logfile = open(os.path.join(self.out_dir, self.COMPILE_LOG_FILE), 'r')
            return logfile.read()
        finally:
            try:
                logfile.close()
            except:
                pass

    def _ExecForRun(self, args, cwd, input, output, timeout):
        try:
            devnull = open(os.devnull, 'w')
            infile = open(input, 'r')
            outfile = open(output, 'w')
            return self._ExecInternal(
                args=args, cwd=cwd,
                stdin=infile, stdout=outfile, stderr=devnull, timeout=timeout)
        finally:
            try:
                devnull.close()
                infile.close()
                outfile.close()
            except:
                pass

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


class CCode(Code):

    def __init__(self, src_name, src_dir, out_dir, flags):
        super(CCode, self).__init__(
            src_name=src_name, src_dir=src_dir, out_dir=out_dir)
        self.exe_name = os.path.splitext(src_name)[0] + FileNames.EXE_EXT
        self.compile_args = ['gcc',
                             '-o', os.path.join(out_dir, self.exe_name),
                             src_name] + flags
        self.run_args = [os.path.join(self.out_dir, self.exe_name)]


class CXXCode(Code):

    def __init__(self, src_name, src_dir, out_dir, flags):
        super(CCode, self).__init__(
            src_name=src_name, src_dir=src_dir, out_dir=out_dir)
        self.exe_name = os.path.splitext(src_name)[0] + FileNames.EXE_EXT
        self.compile_args = ['g++',
                             '-o', os.path.join(out_dir, self.exe_name),
                             src_name] + flags
        self.run_args = [os.path.join(self.out_dir, self.exe_name)]


class JavaCode(Code):

    def __init__(self, src_name, src_dir, out_dir,
                 encoding, mainclass, compile_flags, run_flags):
        super(JavaCode, self).__init__(
            src_name=src_name, src_dir=src_dir, out_dir=out_dir)
        self.encoding = encoding
        self.mainclass = mainclass
        self.compile_args = (['javac', '-d', out_dir] +
                             compile_flags + [src_name])
        self.run_args = (['java', '-Dline.separator=\n', '-cp', self.out_dir] +
                         run_flags + [mainclass])


class ScriptCode(Code):

    QUIET_COMPILE = True

    def __init__(self, src_name, src_dir, out_dir, interpreter):
        super(ScriptCode, self).__init__(
            src_name=src_name, src_dir=src_dir, out_dir=out_dir)
        self.interpreter = interpreter
        self.run_args = [interpreter, os.path.join(self.out_dir, self.src_name)]

    def Compile(self):
        self.MakeOut_Dir()
        FileUtil.CopyFile(os.path.join(self.src_dir, self.src_name),
                          os.path.join(self.out_dir, self.src_name))
        result = RunResult(RunResult.OK, 0.0)
        return (result, "")


class DiffCode(Code):

    QUIET_COMPILE = True

    def __init__(self):
        super(DiffCode, self).__init__(
            src_name="diff", src_dir=None, out_dir=None)

    def Compile(self):
        result = RunResult(RunResult.OK, 0.0)
        return (result, "")

    def Run(self, args, cwd, input, output, timeout):
        parser = OptionParser()
        parser.add_option('-i', '--infile', dest='infile')
        parser.add_option('-d', '--difffile', dest='difffile')
        parser.add_option('-o', '--outfile', dest='outfile')
        (options, pos_args) = parser.parse_args([''] + args)
        run_args = ['diff', '-u', options.difffile, options.outfile]
        return self._ExecForRun(
            args=run_args, cwd=cwd,
            input=input, output=output, timeout=timeout)


class ConfigurableObject(object):

    class Config(object):
        pass

    @classmethod
    def CanLoadFrom(cls, base_dir):
        return os.path.isfile(os.path.join(base_dir, cls.CONFIG_FILE))

    def __init__(self, name, base_dir, parent):
        self.name = name
        self.base_dir = base_dir
        self.parent = parent
        if name is None:
            self.fullname = None
        elif parent is None or parent.fullname is None:
            self.fullname = name
        else:
            self.fullname = parent.fullname + "/" + name
        self.config_file = os.path.join(base_dir, self.CONFIG_FILE)
        real_config_file = self.config_file
        if not os.path.isfile(real_config_file):
            real_config_file = os.devnull
        self.config = ConfigurableObject.Config()
        self.__export_dict = dict()
        self._PreLoad()
        for name in dir(self):
            try:
                attr = getattr(self, name)
                if attr.im_func.__export_mark:
                    self.__export_dict[attr.im_func.func_name] = attr
            except:
                pass
        try:
            f = open(real_config_file, 'rb')
            script = f.read()
            code = compile(script, self.config_file, 'exec')
            exec(code, self.__export_dict, self.config.__dict__)
        finally:
            try:
                f.close()
            except:
                pass
        self._PostLoad()

    def _PreLoad(self):
        pass

    def _PostLoad(self):
        pass

    def SetCacheStamp(self):
        return FileUtil.Touch(self.stamp_file)

    def GetCacheStamp(self):
        if not os.path.isfile(self.stamp_file):
            return None
        return FileUtil.GetModified(self.stamp_file)

    def IsBuildCached(self):
        stamp_mtime = self.GetCacheStamp()
        if stamp_mtime is None:
            return False
        if not os.path.isdir(self.src_dir):
            return False
        for name in ['.'] + os.listdir(self.src_dir):
            if (FileUtil.GetModified(os.path.join(self.src_dir, name)) >
                stamp_mtime):
                return False
        return True

    @classmethod
    def export(cls, f):
        f.__export_mark = True
        return f

    def _AddCodeRegisterer(self, field_name, command_name):
        multiple = (type(getattr(self, field_name)) is list)
        def GenericRegister(code):
            field = getattr(self, field_name)
            if not multiple:
                if field is not None:
                    all_results.Error(self,
                                      "Multiple %ss specified" % command_name)
                    return
                setattr(self, field_name, code)
            if multiple:
                field.append(code)
        @ConfigurableObject.export
        def CRegister(src, flags=['-Wall', '-g', '-O2', '-lm']):
            GenericRegister(CCode(
                src_name=src,
                src_dir=self.src_dir, out_dir=self.out_dir,
                flags=flags))
        @ConfigurableObject.export
        def CXXRegister(src, flags=['-Wall', '-g', '-O2']):
            GenericRegister(CXXCode(
                src_name=src,
                src_dir=self.src_dir, out_dir=self.out_dir,
                flags=flags))
        @ConfigurableObject.export
        def JavaRegister(
            src, encoding='UTF-8', mainclass='Main',
            compile_flags=[], run_flags=['-Xmx256M']):
            GenericRegister(JavaCode(
                src_name=src,
                src_dir=self.src_dir, out_dir=self.out_dir,
                encoding=encoding, mainclass=mainclass,
                compile_flags=compile_flags,
                run_flags=run_flags))
        @ConfigurableObject.export
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
            self.__export_dict[name] = func
            setattr(self, name, func)

    def Build(self):
        raise NotImplementedError()

    def Test(self):
        raise NotImplementedError()

    def Clean(self):
        raise NotImplementedError()

    def Wiki(self):
        raise NotImplementedError()



class RimeRoot(ConfigurableObject):

    CONFIG_FILE = FileNames.RIMEROOT_FILE

    def _PreLoad(self):
        self.root = self

    def _PostLoad(self):
        self.problems = []
        for name in sorted(os.listdir(self.base_dir)):
            dir = os.path.join(self.base_dir, name)
            if Problem.CanLoadFrom(dir):
                problem = Problem(name, dir, self)
                self.problems.append(problem)

    def FindByBaseDir(self, dir):
        if self.base_dir == dir:
            return self
        for problem in self.problems:
            found = problem.FindByBaseDir(dir)
            if found:
                return found
        return None

    def Build(self):
        success = True
        for problem in self.problems:
            if not problem.Build():
                success = False
        return success

    def Test(self):
        success = True
        for problem in self.problems:
            if not problem.Test():
                success = False
        return success

    def Clean(self):
        success = True
        for problem in self.problems:
            if not problem.Clean():
                success = False
        return success

    def Wiki(self):
        self.Test()
        wiki = PukiWikiGenerator.Generate()
        print wiki
        return True

            

class Problem(ConfigurableObject):

    CONFIG_FILE = FileNames.PROBLEM_FILE

    def _PreLoad(self):
        self.root = self.parent
        self.out_dir = os.path.join(self.base_dir, FileNames.RIME_OUT_DIR)

    def _PostLoad(self):
        if not hasattr(self.config, 'TIME_LIMIT'):
            all_results.Error(self, "Time limit is not specified")
        else:
            self.timeout = self.config.TIME_LIMIT
        self.solutions = []
        for name in sorted(os.listdir(self.base_dir)):
            dir = os.path.join(self.base_dir, name)
            if Solution.CanLoadFrom(dir):
                solution = Solution(name, dir, self)
                self.solutions.append(solution)
        self._DetermineReferenceSolution()
        self.tests = Tests(
            FileNames.TESTS_DIR,
            os.path.join(self.base_dir, FileNames.TESTS_DIR),
            self)

    def _DetermineReferenceSolution(self):
        self.reference_solution = None
        if not hasattr(self.config, 'REFERENCE_SOLUTION'):
            for solution in self.solutions:
                if solution.IsCorrect():
                    self.reference_solution = solution
                    break
        else:
            reference_solution_name = self.config.REFERENCE_SOLUTION
            for solution in self.solutions:
                if solution.name == reference_solution_name:
                    self.reference_solution = solution
                    break
            if self.reference_solution is None:
                all_results.Error(
                    self,
                    ("Reference solution \"%s\" does not exist" %
                     reference_solution_name))

    def FindByBaseDir(self, dir):
        if self.base_dir == dir:
            return self
        for solution in self.solutions:
            found = solution.FindByBaseDir(dir)
            if found:
                return found
        return None

    def Build(self):
        success = True
        for solution in self.solutions:
            if not solution.Build():
                success = False
        if not self.tests.Build():
            success = False
        return success

    def Test(self):
        return self.tests.Test()

    def Clean(self):
        Console.PrintAction("CLEAN", self)
        success = True
        if not self.tests.Clean():
            success = False
        for solution in self.solutions:
            if not solution.Clean():
                success = False
        if not FileUtil.RemoveTree(self.out_dir):
            success = False
        return success


class Tests(ConfigurableObject):

    CONFIG_FILE = FileNames.TESTS_FILE

    def _PreLoad(self):
        self.problem = self.parent
        self.root = self.parent.root
        self.src_dir = self.base_dir
        self.out_dir = os.path.join(self.problem.out_dir, FileNames.TESTS_DIR)
        self.stamp_file = os.path.join(self.out_dir, FileNames.STAMP_FILE)
        self.generators = []
        self.validator = None
        self.judge = None
        self._AddCodeRegisterer('generators', 'generator')
        self._AddCodeRegisterer('validator', 'validator')
        self._AddCodeRegisterer('judge', 'judge')
        if not os.path.isfile(self.config_file):
            all_results.Warning(self,
                                "%s file does not exist" % self.CONFIG_FILE)

    def _PostLoad(self):
        # TODO: print warnings if no validator / judge is specified.
        if self.judge is None:
            self.judge = DiffCode()
        pass

    def FindByBaseDir(self, dir):
        if self.base_dir == dir:
            return self
        return None

    def Build(self):
        #Console.PrintAction("BUILD", self)
        if self.IsBuildCached():
            #Console.PrintAction("BUILD", self, "(cached)", overwrite=True)
            return True
        if not FileUtil.RemoveTree(self.out_dir):
            all_results.Error(self,
                              "Failed to remove output dir")
            return False
        if not os.path.isdir(self.src_dir):
            if not FileUtil.MakeDir(self.out_dir):
                all_results.Error(self,
                                  "Failed to create output dir")
                return False
        else:
            if not FileUtil.CopyTree(self.src_dir, self.out_dir):
                all_results.Error(self,
                                  "Failed to create output dir")
                return False
        if not self._CompileGenerator():
            return False
        if not self._CompileValidator():
            return False
        if not self._CompileJudge():
            return False
        if not self._RunGenerator():
            return False
        if not self._RunValidator():
            return False
        if self._ListInputFiles():
            if not self._CompileReferenceSolution():
                return False
            if not self._RunReferenceSolution():
                return False
        if not self.SetCacheStamp():
            all_results.Error(self,
                              "Failed to create stamp file")
            return False
        return True

    def _CompileGenerator(self):
        for generator in self.generators:
            if not generator.QUIET_COMPILE:
                Console.PrintAction("COMPILE", self, generator.src_name)
            (res, log) = generator.Compile()
            if res.status != RunResult.OK:
                all_results.Error(self,
                                  "%s: Compile Error" % generator.src_name)
                Console.PrintLog(log)
                return False
        return True

    def _RunGenerator(self):
        for generator in self.generators:
            Console.PrintAction("GENERATE", self, generator.src_name)
            res = generator.Run(
                args=[], cwd=self.out_dir,
                input=os.devnull, output=os.devnull, timeout=None)
            if res.status != RunResult.OK:
                all_results.Error(
                    self, generator.src_name, "Runtime Error")
                return False
        return True

    def _CompileValidator(self):
        if self.validator is None:
            return True
        if not self.validator.QUIET_COMPILE:
            Console.PrintAction("COMPILE", self, self.validator.src_name)
        (res, log) = self.validator.Compile()
        if res.status != RunResult.OK:
            results.Error(self,
                          "%s: Compile Error" % self.validator.src_name)
            Console.PrintLog(log)
            return False
        return True

    def _RunValidator(self):
        Console.PrintAction("VALIDATE", self)
        infiles = self._ListInputFiles()
        test_result = TestResult(None, None, infiles)
        for (i, infile) in enumerate(infiles):
            test_result.SetStatus(infile, TestResult.RUN)
            if RimeOptions.use_indicator:
                Console.PrintAction(
                    "VALIDATE", self,
                    "%s [%s]" % (test_result.GetIndicator(), infile),
                    overwrite=True)
            else:
                Console.PrintAction(
                    "VALIDATE", self,
                    "[%d/%d] %s" % (i+1, len(infiles), infile),
                    overwrite=True)
            res = self.validator.Run(
                args=[], cwd=self.out_dir,
                input=os.path.join(self.out_dir, infile), output=os.devnull,
                timeout=None)
            if res.status == RunResult.NG:
                results.Error(self, self.validator.src_name,
                              "Validation Failed")
                return False
            elif res.status != RunResult.OK:
                results.Error(self, self.validator.src_name,
                              "Runtime Error on %s" % infile)
                return False
            test_result.SetStatus(infile, TestResult.AC)
        if RimeOptions.use_indicator:
            Console.PrintAction(
                "VALIDATE", self, test_result.GetIndicator(),
                overwrite=True)
        else:
            Console.PrintAction(
                "VALIDATE", self, TestResult.PASSED,
                overwrite=True)
        return True

    def _CompileJudge(self):
        if self.judge is None:
            return True
        if not self.judge.QUIET_COMPILE:
            Console.PrintAction("COMPILE",
                                self,
                                self.judge.src_name)
        (res, log) = self.judge.Compile()
        if res.status != RunResult.OK:
            results.Error(self,
                          "%s: Compile Error" % self.judge.src_name)
            Console.PrintLog(log)
            return False
        return True

    def _CompileReferenceSolution(self):
        reference_solution = self.problem.reference_solution
        if reference_solution is None:
            all_results.Error(self,
                              "Reference solution is not available")
            return False
        return reference_solution.Build()

    def _RunReferenceSolution(self):
        reference_solution = self.problem.reference_solution
        if reference_solution is None:
            all_results.Error(self,
                              "Reference solution is not available")
            return False
        Console.PrintAction("REFRUN", reference_solution)
        infiles = self._ListInputFiles()
        test_result = TestResult(None, None, infiles)
        for (i, infile) in enumerate(infiles):
            difffile = os.path.splitext(infile)[0] + FileNames.DIFF_EXT
            if os.path.isfile(os.path.join(self.out_dir, difffile)):
                test_result.SetStatus(infile, TestResult.AC)
                continue
            test_result.SetStatus(infile, TestResult.RUN)
            if RimeOptions.use_indicator:
                Console.PrintAction(
                    "REFRUN", reference_solution,
                    "%s [%s]" % (test_result.GetIndicator(), infile),
                    overwrite=True)
            else:
                Console.PrintAction(
                    "REFRUN", reference_solution,
                    "[%d/%d] %s" % (i+1, len(infiles), infile),
                    overwrite=True)
            res = reference_solution.Run(
                args=[], cwd=self.out_dir,
                input=os.path.join(self.out_dir, infile),
                output=os.path.join(self.out_dir, difffile),
                timeout=None)
            if res.status != RunResult.OK:
                all_results.Error(reference_solution,
                                  "Runtime Error on %s" % infile)
                return False
            test_result.SetStatus(infile, TestResult.AC)
        if RimeOptions.use_indicator:
            Console.PrintAction(
                "REFRUN", reference_solution, test_result.GetIndicator(),
                overwrite=True)
        else:
            Console.PrintAction(
                "REFRUN", reference_solution,
                overwrite=True)
        return True

    def Test(self, solution=None):
        if not self.Build():
            return False
        if solution is not None:
            return self._TestOneSolution(solution)
        success = True
        for solution in self.problem.solutions:
            if not self._TestOneSolution(solution):
                success = False
        return success

    def _TestOneSolution(self, solution):
        if not solution.Build():
            test_result = TestResult(self.problem, solution, [])
            test_result.result = TestResult.FAILED
            test_result.detail = "Compile Error"
            all_results.AddTestResult(test_result)
            return False
        cookie = solution.GetCacheStamp()
        Console.PrintAction("TEST", solution)
        infiles = self._ListInputFiles()
        test_result = TestResult(self.problem, solution, infiles)
        test_result.result = TestResult.PASSED
        error = None
        any_cached = False
        if not solution.IsCorrect() and solution.challenge_cases:
            challenge_cases = self._SortInputFiles(solution.challenge_cases)
            all_exists = True
            for infile in challenge_cases:
                if infile not in infiles:
                    results.Error(solution,
                                      "Challenge case not found: %s" % infile)
                    all_exists = False
            if not all_exists:
                return False
            for (i, infile) in enumerate(challenge_cases):
                test_result.SetStatus(infile, TestResult.RUN)
                if RimeOptions.use_indicator:
                    Console.PrintAction(
                        "TEST", solution,
                        "%s [%s]" % (test_result.GetIndicator(), infile),
                        overwrite=True)
                else:
                    Console.PrintAction(
                        "TEST", solution,
                        "[%d/%d] %s" % (i+1, len(infiles), infile),
                        overwrite=True)
                (status, time, cached) = self._TestOneCase(
                    solution, infile, cookie)
                if cached:
                    any_cached = True
                test_result.SetStatus(infile, status)
                if status == TestResult.ERR:
                    all_results.Error(solution,
                                      "Validation Error on %s" % infile,
                                      quiet=True)
                    test_result.ruling_file = infile
                    test_result.result = TestResult.FAILED
                    break
                if status == TestResult.AC:
                    all_results.Error(solution,
                                      "Unexpectedly Accepted for %s" % infile,
                                      quiet=True)
                    test_result.ruling_file = infile
                    test_result.result = TestResult.FAILED
                    break
        else:
            for (i, infile) in enumerate(infiles):
                test_result.SetStatus(infile, TestResult.RUN)
                if RimeOptions.use_indicator:
                    Console.PrintAction(
                        "TEST", solution,
                        "%s [%s]" % (test_result.GetIndicator(), infile),
                        overwrite=True)
                else:
                    Console.PrintAction(
                        "TEST", solution,
                        "[%d/%d] %s" % (i+1, len(infiles), infile),
                        overwrite=True)
                (status, time, cached) = self._TestOneCase(
                    solution, infile, cookie)
                if cached:
                    any_cached = True
                test_result.SetStatus(infile, status)
                if status == TestResult.ERR:
                    all_results.Error(solution,
                                      "Validation Error on %s" % infile,
                                      quiet=True)
                    test_result.ruling_file = infile
                    test_result.result = TestResult.FAILED
                    break
                if status != TestResult.AC:
                    if not solution.IsCorrect():
                        break
                    all_results.Error(
                        solution,
                        "%s on %s" % (test_result.GetRejectReason(infile),
                                      infile),
                        quiet=True)
                    test_result.ruling_file = infile
                    test_result.result = TestResult.FAILED
                    break
                test_result.SetTime(infile, time)
            if not solution.IsCorrect() and test_result.IsAllAccepted():
                test_result.result = TestResult.FAILED
        status_line = ""
        if RimeOptions.use_indicator:
            status_line += test_result.GetIndicator()
            if test_result.ruling_file:
                status_line += " [%s]" % test_result.ruling_file
        else:
            if test_result.result == TestResult.PASSED:
                status_line += TestResult.PASSED
                if test_result.HasTimeStats():
                    status_line += " (%s)" % test_result.GetTimeStats()
            else:
                if test_result.ruling_file:
                    status_line += ("\x1b[31m%s\x1b[0m: \x1b[1m%s\x1b[0m" %
                                    (test_result.GetRejectReason(
                                         test_result.ruling_file),
                                     test_result.ruling_file))
                else:
                    status_line += "Unexpectedly Accepted"
        if any_cached:
            status_line += " (cached)"
        Console.PrintAction("TEST", solution, status_line, overwrite=True)
        all_results.AddTestResult(test_result)
        return (test_result.result == TestResult.PASSED)

    def _TestOneCase(self, solution, infile, cookie):
        cachefile = os.path.join(
            solution.out_dir,
            os.path.splitext(infile)[0] + FileNames.CACHE_EXT)
        if RimeOptions.cache_tests:
            if cookie is not None and os.path.isfile(cachefile):
                (cached_cookie, result) = FileUtil.PickleLoad(cachefile)
                if cached_cookie == cookie:
                    return tuple(list(result)+[True])
        result = self._TestOneCaseNoCache(solution, infile)
        FileUtil.PickleSave((cookie, result), cachefile)
        return tuple(list(result)+[False])

    def _TestOneCaseNoCache(self, solution, infile):
        outfile = os.path.splitext(infile)[0] + FileNames.OUT_EXT
        difffile = os.path.splitext(infile)[0] + FileNames.DIFF_EXT
        judgefile = os.path.splitext(infile)[0] + FileNames.JUDGE_EXT
        res = solution.Run(
            args=[], cwd=solution.out_dir,
            input=os.path.join(self.out_dir, infile),
            output=os.path.join(solution.out_dir, outfile),
            timeout=self.problem.timeout)
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
            input=os.devnull, output=os.path.join(solution.out_dir, judgefile),
            timeout=None)
        if res.status == RunResult.RE:
            return (TestResult.ERR, None)
        if res.status != RunResult.OK:
            return (TestResult.WA, None)
        return (TestResult.AC, time)

    def Clean(self):
        Console.PrintAction("CLEAN", self)
        return FileUtil.RemoveTree(self.out_dir)

    def _ListInputFiles(self):
        infiles = [s for s in os.listdir(self.out_dir)
                   if s.endswith(FileNames.IN_EXT)]
        return self._SortInputFiles(infiles)

    def _SortInputFiles(self, infiles):
        infiles = infiles[:]
        def tokenize_cmp(a, b):
            def tokenize(s):
                def replace_digits(match):
                    return "%08s" % match.group(0)
                return re.sub(r'\d+', replace_digits, s)
            return cmp(tokenize(a), tokenize(b))
        infiles.sort(tokenize_cmp)
        return infiles



class Solution(ConfigurableObject):

    CONFIG_FILE = FileNames.SOLUTION_FILE

    def _PreLoad(self):
        self.problem = self.parent
        self.root = self.parent.root
        self.src_dir = self.base_dir
        self.out_dir = os.path.join(self.problem.out_dir, self.name)
        self.stamp_file = os.path.join(self.out_dir, FileNames.STAMP_FILE)
        self.code = None
        self._AddCodeRegisterer('code', 'solution')

    def _PostLoad(self):
        source_exts = {
            '.c': self.c_solution,
            '.cc': self.cxx_solution,
            '.cpp': self.cxx_solution,
            '.java': self.java_solution,
            }
        if self.code is None:
            src = None
            solution_func = None
            ambiguous = False
            for name in os.listdir(self.src_dir):
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
                all_results.Error(self,
                                  ("Multiple source files found; " +
                                   "specify explicitly in " +
                                   self.CONFIG_FILE))
            elif src is None:
                all_results.Error(self, "Source file not found")
            else:
                solution_func(src=src)
        if hasattr(self.config, 'CHALLENGE_CASES'):
            self.correct = False
            self.challenge_cases = self.config.CHALLENGE_CASES
        else:
            self.correct = True
            self.challenge_cases = None

    def FindByBaseDir(self, dir):
        if self.base_dir == dir:
            return self
        return None

    def IsCorrect(self):
        return self.correct

    def Build(self):
        #Console.PrintAction("BUILD", self)
        if self.IsBuildCached():
            #Console.PrintAction("BUILD", self, "(cached)", overwrite=True)
            return True
        if not self.code.QUIET_COMPILE:
            Console.PrintAction("COMPILE", self)
        (res, log) = self.code.Compile()
        if res.status != RunResult.OK:
            all_results.Error(self, "Compile Error")
            Console.PrintLog(log)
            return False
        if log:
            all_results.Warning(self, "Compiler warnings found")
            Console.PrintLog(log)
        if not self.SetCacheStamp():
            all_results.Error(self,
                              "Failed to create stamp file")
            return False
        return True

    def Test(self):
        return self.problem.tests.Test(self)

    def Run(self, args, cwd, input, output, timeout):
        return self.code.Run(args=args, cwd=cwd,
                             input=input, output=output, timeout=timeout)

    def Clean(self):
        Console.PrintAction("CLEAN", self)
        return self.code.Clean()


class PukiWikiGenerator(object):

    CELL_GOOD = "BGCOLOR(#ccffcc):○"
    CELL_NEUTRAL = "BGCOLOR(#ffffcc):△"
    CELL_BAD = "BGCOLOR(#ffcccc):×"
    CELL_NA = "BGCOLOR(#cccccc):－"

    @classmethod
    def Generate(cls):
        test_results = sorted(all_results.test_results, TestResult.Compare)
        wiki = "// Generated by Rime\n"
        wiki += "|Problem|Solution||Reason|h\n"
        last_problem = None
        for test_result in test_results:
            if last_problem is test_result.problem:
                wiki += "|^"
            else:
                wiki += "|%s" % test_result.problem.name
            if test_result.solution.IsCorrect():
                wiki += "|%s" % test_result.solution.name
                if test_result.solution is test_result.problem.reference_solution:
                    wiki += "(*)"
            else:
                wiki += "|&color(#888888){%s};" % test_result.solution.name
            if test_result.result == TestResult.PASSED:
                wiki += "|%s" % cls.CELL_GOOD
            else:
                wiki += "|%s" % cls.CELL_BAD
            if test_result.detail:
                wiki += "|%s" % test_result.detail
            elif test_result.result == TestResult.FAILED:
                ruling_file = test_result.ruling_file
                reason = test_result.GetRejectReason(ruling_file)
                wiki += "|%s (%s)" % (reason, ruling_file)
            else:
                wiki += "|"
            wiki += "|\n"
            last_problem = test_result.problem
        return wiki


class Rime(object):

    COMMANDS = ['build', 'test', 'clean']

    def Main(self, args):
        Console.Print("Rime: Tool for Programming Contest Organizers")
        Console.Print("")
        self._ParseArgs(args)
        self._LoadConfig()
        if all_results.HasError():
            Console.PrintTitle("SUMMARY")
            all_results.PrintErrorSummary()
            sys.exit(1)
        if self.cmd is None:
            Console.PrintError("No command specified")
            return
        if self.cmd not in self.COMMANDS:
            Console.PrintError("Unknown command: %s" % self.cmd)
            sys.exit(1)
        method_name = self.cmd.title()
        method = getattr(self.target, method_name)
        try:
            method()
        except NotImplementedError:
            Console.PrintError("Command %s is not applicable here" % self.cmd)
            sys.exit(1)
        Console.PrintTitle("SUMMARY")
        all_results.PrintTestSummary()
        all_results.PrintErrorSummary()

    def _ParseArgs(self, args):
        parser = OptionParser(usage="%prog command [dir] [options]")
        parser.add_option('-I', '--indicator', dest='use_indicator',
                          default=False, action="store_true",
                          help="use indicator for progress display")
        parser.add_option('-C', '--cache-tests', dest='cache_tests',
                          default=False, action="store_true",
                          help="cache test results")
        (self.options, self.args) = parser.parse_args(args[1:])
        for name in dir(RimeOptions):
            if name.startswith('_'):
                continue
            setattr(RimeOptions, name, getattr(self.options, name))
        self.cmd = self.args[0] if self.args else None
        if self.cmd == 'help':
            parser.print_help()
            sys.exit(0)
        if len(self.args) <= 1:
            self.target_dir = None
            self.params = []
        else:
            self.target_dir = os.path.abspath(self.args[1])
            self.params = self.args[2:]

    def _LoadConfig(self):
        if self.target_dir is None:
            dir = os.getcwd()
        else:
            dir = self.target_dir
        while not RimeRoot.CanLoadFrom(dir):
            (head, tail) = os.path.split(dir)
            if head == dir:
                Console.PrintError("%s not found" % FileNames.RIMEROOT_FILE)
                sys.exit(1)
            dir = head
        self.root = RimeRoot(None, dir, None)
        if self.target_dir is None:
            self.target_dir = self.root.base_dir
        self.target = self.root.FindByBaseDir(self.target_dir)
        if self.target is None:
            all_results.Error(None,
                              "Specified directory is not maintained by Rime")


if __name__ == '__main__':
    rime = Rime()
    rime.Main(sys.argv)

