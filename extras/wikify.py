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


BGCOLOR_TITLE  = 'BGCOLOR(#eeeeee):'
BGCOLOR_NORMAL = 'BGCOLOR(#f8f8f8):'
BGCOLOR_GOOD   = 'BGCOLOR(#ccffcc):'
BGCOLOR_NOTBAD = 'BGCOLOR(#ffffcc):'
BGCOLOR_BAD    = 'BGCOLOR(#ffcccc):'
BGCOLOR_NA     = 'BGCOLOR(#cccccc):'

CELL_GOOD   = BGCOLOR_GOOD + '&#x25cb;'
CELL_NOTBAD = BGCOLOR_NOTBAD + '&#x25b3;'
CELL_BAD    = BGCOLOR_BAD + '&#xd7;'
CELL_NA     = BGCOLOR_NA + '-'



def LoadRimeModule():
  dir = os.getcwd()
  while not os.path.isfile(os.path.join(dir, 'rime.py')):
    pardir = os.path.dirname(dir)
    if dir == pardir:
      raise Exception("rime.py not found!")
    dir = pardir
  global rime
  rime = imp.load_source('rime', os.path.join(dir, 'rime.py'))


def GenerateWiki(root, errors):
  # Clean and update.
  root.Clean(errors)
  rime.Console.PrintAction("UPDATE", None, "svn up")
  subprocess.call(['svn', 'up'])
  # Get system information.
  rev = commands.getoutput('svnversion')
  username = getpass.getuser()
  hostname = socket.gethostname()
  # Generate content.
  wiki = "このセクションは wikify.py により自動生成されています (rev.%s, uploaded by %s @ %s)\n" % (rev, username, hostname)
  wiki += "|||CENTER:|CENTER:|CENTER:|CENTER:|CENTER:|c\n"
  wiki += "|~問題|~担当|~解答|~入力|~出力|~入検|~出検|\n"
  for problem in root.problems:
    # For each problem fill these cells.
    cell_solutions = CELL_BAD
    cell_input = CELL_BAD
    cell_output = CELL_BAD
    cell_validator = CELL_BAD
    cell_judge = CELL_NA
    # Get status.
    title = problem.config.get('TITLE') or "No Title"
    wikiname = problem.config.get('WIKI_NAME') or "No Wiki Name"
    assignees = problem.config.get('ASSIGNEES') or ""
    if type(assignees) is list:
      assignees = ",".join(assignees)
    results = problem.Test(errors)
    num_tests = len(problem.tests.ListInputFiles())
    correct_solution_results = [result for result in results if result.solution.IsCorrect()]
    num_corrects = len(correct_solution_results)
    num_bads = len([result for result in results if not result.good])
    input_fixed = problem.config.get('INPUT_FIXED')
    need_custom_judge = problem.config.get('NEED_CUSTOM_JUDGE')
    # Solutions:
    if num_corrects >= 2:
      cell_solutions = CELL_GOOD
    elif num_corrects >= 1:
      cell_solutions = CELL_NOTBAD
    # Input:
    if input_fixed:
      cell_input = BGCOLOR_GOOD + str(num_tests)
    elif num_tests >= 20:
      cell_input = BGCOLOR_NOTBAD + str(num_tests)
    # Output:
    if num_bads == 0 and num_tests >= 10 and num_corrects >= 2:
      cell_output = CELL_GOOD
    # Validator:
    if problem.tests.validator is not None:
      cell_validator = CELL_GOOD
    # Judge:
    if need_custom_judge:
      if problem.tests.judge.__class__.__name__ != "DiffCode":
        cell_judge = CELL_GOOD
      else:
        cell_judge = CELL_BAD
    # Done.
    wiki += "|[[%(title)s>%(wikiname)s]]|%(assignees)s|%(cell_solutions)s|%(cell_input)s|%(cell_output)s|%(cell_validator)s|%(cell_judge)s|\n" % locals()
  return wiki


def UploadWiki(root, wiki):
  wiki_url = root.config['WIKI_URL']
  page_name = root.config['WIKI_STATUS_PAGE']
  wiki_encoding = root.config['WIKI_ENCODING']
  auth_realm = root.config['WIKI_AUTH_REALM']
  auth_hostname = root.config['WIKI_AUTH_HOSTNAME']
  auth_username = root.config['WIKI_AUTH_USERNAME']
  auth_password = root.config['WIKI_AUTH_PASSWORD']
  rime.Console.PrintAction("UPLOAD", None, wiki_url)
  auth_handler = urllib2.HTTPBasicAuthHandler()
  auth_handler.add_password(auth_realm, auth_hostname, auth_username, auth_password)
  opener = urllib2.build_opener(auth_handler)
  urllib2.install_opener(opener)
  native_page_name = unicode(page_name, 'utf8').encode(wiki_encoding)
  edit_page = urllib2.urlopen("%s?cmd=edit&page=%s" % (wiki_url, urllib.quote(native_page_name))).read()
  params = dict(
    cmd='edit',
    page=page_name,
    digest=re.search(r'value="([0-9a-f]{32})"', edit_page).group(1),
    msg=wiki,
    write="ページの更新",
    encode_hint="ぷ")
  urllib2.urlopen(wiki_url, urllib.urlencode(params))


def main():
  # Initialize Rime object.
  LoadRimeModule()
  errors = rime.ErrorRecorder()
  arime = rime.Rime()
  options = arime.GetDefaultOptions()
  root = arime.LoadRoot(os.getcwd(), options, errors)
  if not root:
    errors.PrintSummary()
    return
  os.chdir(root.base_dir)
  wiki = GenerateWiki(root, errors)
  UploadWiki(root, wiki)


if __name__ == '__main__':
  main()
