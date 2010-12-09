#!/bin/bash

cd "$(dirname "$0")/.."

for test in test/test_*.py; do
    echo $test
    python $test || exit $?
    echo
done
