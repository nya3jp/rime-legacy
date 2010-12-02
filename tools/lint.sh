#!/bin/sh

cd "$(dirname "$0")"/..
./tools/pychecker/checker.py --limit=100000 --no-argsused rime.py
