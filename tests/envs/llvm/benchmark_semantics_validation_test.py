# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
import re
import tempfile
from pathlib import Path
from typing import Set

import pytest

from compiler_gym.envs import LlvmEnv
from compiler_gym.envs.llvm import datasets
from compiler_gym.envs.llvm.datasets import get_llvm_benchmark_validation_callback
from tests.test_main import main

pytest_plugins = ["tests.envs.llvm.fixtures"]

# The set of cBench benchmarks which do not have support for semantics
# validation.
CBENCH_VALIDATION_EXCLUDE_LIST: Set[str] = {
    "benchmark://cBench-v0/ispell",
}


def test_no_validation_callback_for_custom_benchmark(env: LlvmEnv):
    """Test that a custom benchmark has no validation callback."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "example.c"
        with open(p, "w") as f:
            print("int main() {return 0;}", file=f)
        benchmark = env.make_benchmark(p)

    env.benchmark = benchmark
    env.reset()

    validation_cb = get_llvm_benchmark_validation_callback(env)

    assert validation_cb is None


def test_expand_command_vars():
    assert datasets.expand_command_vars("abc") == "abc"
    assert datasets.expand_command_vars("$BIN") == "a.out"
    assert re.match(
        r"a.out .+/runtime_data/foo .+/runtime_data/bar",
        datasets.expand_command_vars("$BIN $D/foo $O/bar"),
    )
    assert datasets.expand_command_vars("$UNKNOWN $BIN") == "$UNKNOWN a.out"


def test_get_cBench_reference_output_sha(env: LlvmEnv):
    """SHA benchmark prints 5 random hex strings. Normally these hex strings are
    16 characters but occasionally they are less (presumably becuase of a
    leading zero being omitted).
    """
    output = datasets.get_cBench_reference_output(
        "cBench-v0/sha", "$BIN $D/office_data/1.txt", tuple()
    )
    assert re.match(
        r"[0-9a-f]{0,16} [0-9a-f]{0,16} [0-9a-f]{0,16} [0-9a-f]{0,16} [0-9a-f]{0,16}",
        output.decode("utf-8").rstrip(),
    )


def test_validate_cBench_unoptimized(env: LlvmEnv, benchmark_name: str):
    """Run the validation routine on unoptimized version of all cBench benchmarks."""
    env.reset(benchmark=benchmark_name)
    cb = datasets.get_llvm_benchmark_validation_callback(env)

    if benchmark_name in CBENCH_VALIDATION_EXCLUDE_LIST:
        assert cb is None
    else:
        assert cb
        assert cb(env) is None


if __name__ == "__main__":
    main()
