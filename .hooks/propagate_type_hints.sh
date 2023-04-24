#!/usr/bin/env bash
if [[ -n "$(echo $@ | grep linear_operator/operators)" ]]; then
  python ./.hooks/propagate_type_hints.py
  if [[ $TYPE_HINTS_PROPAGATED = 1 ]]; then
    exit 2
  fi
fi
