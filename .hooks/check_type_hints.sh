#!/usr/bin/env bash
echo "HI" 1>&2
CHANGED_FILES=$(git diff --cached --name-only | grep linear_operator/operators)
echo $CHANGED_FILES 1>&2
if [[ -n "$CHANGED_FILES" ]]; then
  python ./.hooks/propagate_type_hints.py
else
  echo "NO CHANGED FILES" 1>&2
fi
