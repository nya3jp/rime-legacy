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
# Original Author: Hisayori Noda
#

import commands
import getpass
import imp
import os
import re
import socket
import subprocess
import sys
import urllib
import urllib2


HELP_MESSAGE = """\
Usage: wikify.py [OPTIONS]

Options are the same as rime.py.
"""

BGCOLOR_TITLE  = 'BGCOLOR(#eeeeee):'
BGCOLOR_GOOD   = 'BGCOLOR(#ccffcc):'
BGCOLOR_NOTBAD = 'BGCOLOR(#ffffcc):'
BGCOLOR_BAD    = 'BGCOLOR(#ffcccc):'
BGCOLOR_NA     = 'BGCOLOR(#cccccc):'

CELL_GOOD   = BGCOLOR_GOOD + '&#x25cb;'
CELL_NOTBAD = BGCOLOR_NOTBAD + '&#x25b3;'
CELL_BAD    = BGCOLOR_BAD + '&#xd7;'
CELL_NA     = BGCOLOR_NA + '-'

RIME_FILENAME = 'rime.py'


# Extend sys.path to load Rime from non-standard path.
def ExtendPathForRime():
  dir = os.getcwd()
  while not os.path.isfile(os.path.join(dir, RIME_FILENAME)):
    pardir = os.path.dirname(dir)
    if dir == pardir:
      raise Exception(RIME_FILENAME + ' not found!')
    dir = pardir
  sys.path.append(dir)

ExtendPathForRime()

import rime


class Wikify(object):

  def Main(self, rime_obj, ctx):
    root = rime_obj.LoadRoot(os.getcwd(), ctx)
    if not root:
      ctx.errors.PrintSummary()
      return
    graph = rime_obj.CreateTaskGraph(ctx)
    try:
      wiki = graph.Run(self._GenerateWiki(root, ctx))
    finally:
      graph.Close()
    self._UploadWiki(root, wiki)

  @rime.GeneratorTask.FromFunction
  def _GenerateWiki(self, root, ctx):
    # Clean and update.
    yield root.Clean(ctx)
    rime.Console.PrintAction('UPDATE', None, 'svn up')
    subprocess.call(['svn', 'up'])
    # Get system information.
    rev = commands.getoutput('svnversion')
    username = getpass.getuser()
    hostname = socket.gethostname()
    # Generate content.
    wiki = ('このセクションは wikify.py により自動生成されています '
            '(rev.%(rev)s, uploaded by %(username)s @ %(hostname)s)\n' %
            dict(rev=rev, username=username, hostname=hostname))
    wiki += '|||CENTER:|CENTER:|CENTER:|CENTER:|CENTER:|c\n'
    wiki += '|~問題|~担当|~解答|~入力|~出力|~入検|~出検|\n'
    results = yield rime.TaskBranch([
        self._GenerateWikiOne(problem, ctx)
        for problem in root.problems])
    wiki += ''.join(results)
    yield wiki

  @rime.GeneratorTask.FromFunction
  def _GenerateWikiOne(self, problem, ctx):
    # Get status.
    title = problem.config.get('TITLE', 'No Title')
    wikiname = problem.config.get('WIKI_NAME', 'No Wiki Name')
    assignees = problem.config.get('ASSIGNEES', '')
    if type(assignees) is list:
      assignees = ','.join(assignees)
    # Fetch test results.
    results = yield problem.Test(ctx)
    # Get various information about the problem.
    num_solutions = len(results)
    num_tests = len(problem.tests.ListInputFiles())
    correct_solution_results = [result for result in results
                                if result.solution.IsCorrect()]
    num_corrects = len(correct_solution_results)
    num_incorrects = num_solutions - num_corrects
    num_agreed = len([result for result in correct_solution_results
                      if result.good])
    input_fixed = problem.config.get('INPUT_FIXED')
    need_custom_judge = problem.config.get('NEED_CUSTOM_JUDGE')
    # Solutions:
    if num_corrects >= 2:
      cell_solutions = BGCOLOR_GOOD
    elif num_corrects >= 1:
      cell_solutions = BGCOLOR_NOTBAD
    else:
      cell_solutions = BGCOLOR_BAD
    cell_solutions += '%d+%d' % (num_corrects, num_incorrects)
    # Input:
    if input_fixed:
      cell_input = BGCOLOR_GOOD + str(num_tests)
    elif num_tests >= 20:
      cell_input = BGCOLOR_NOTBAD + str(num_tests)
    else:
      cell_input = BGCOLOR_BAD + str(num_tests)
    # Output:
    if num_corrects >= 2 and num_agreed == num_corrects:
      cell_output = BGCOLOR_GOOD
    elif num_agreed >= 2:
      cell_output = BGCOLOR_NOTBAD
    else:
      cell_output = BGCOLOR_BAD
    cell_output += '%d/%d' % (num_agreed, num_corrects)
    # Validator:
    if problem.tests.validators:
      cell_validator = CELL_GOOD
    else:
      cell_validator = CELL_BAD
    # Judge:
    if need_custom_judge:
      if problem.tests.judge.__class__.__name__ != 'DiffCode':
        cell_judge = CELL_GOOD
      else:
        cell_judge = CELL_BAD
    else:
      cell_judge = CELL_NA
    # Done.
    yield (('|[[%(title)s>%(wikiname)s]]|%(assignees)s|'
            '%(cell_solutions)s|%(cell_input)s|%(cell_output)s|'
            '%(cell_validator)s|%(cell_judge)s|\n') % locals())

  def _UploadWiki(self, root, wiki):
    wiki_url = root.config['WIKI_URL']
    page_name = root.config['WIKI_STATUS_PAGE']
    wiki_encoding = root.config['WIKI_ENCODING']
    auth_realm = root.config['WIKI_AUTH_REALM']
    auth_hostname = root.config['WIKI_AUTH_HOSTNAME']
    auth_username = root.config['WIKI_AUTH_USERNAME']
    auth_password = root.config['WIKI_AUTH_PASSWORD']
    rime.Console.PrintAction('UPLOAD', None, wiki_url)
    auth_handler = urllib2.HTTPBasicAuthHandler()
    auth_handler.add_password(auth_realm, auth_hostname, auth_username, auth_password)
    opener = urllib2.build_opener(auth_handler)
    urllib2.install_opener(opener)
    native_page_name = unicode(page_name, 'utf8').encode(wiki_encoding)
    edit_page = urllib2.urlopen('%s?cmd=edit&page=%s' % (wiki_url, urllib.quote(native_page_name))).read()
    params = dict(
      cmd='edit',
      page=page_name,
      digest=re.search(r'value="([0-9a-f]{32})"', edit_page).group(1),
      msg=wiki,
      write='ページの更新',
      encode_hint='ぷ')
    urllib2.urlopen(wiki_url, urllib.urlencode(params))


def main():
  rime_obj = rime.Rime()
  option_parser = rime_obj.GetOptionParser()
  options, extra_args = option_parser.parse_args(sys.argv)
  if options.show_help:
    print HELP_MESSAGE
    return
  ctx = rime.RimeContext(options)
  wikify = Wikify()
  wikify.Main(rime_obj, ctx)


if __name__ == '__main__':
  main()
