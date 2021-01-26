# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines the available LLVM datasets."""
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import gym

from compiler_gym.datasets.dataset import Dataset
from compiler_gym.util.runfiles_path import runfiles_path
from compiler_gym.util.temporary_working_directory import temporary_working_directory
from compiler_gym.util.timer import Timer

_CLANG = runfiles_path("CompilerGym/compiler_gym/third_party/llvm/clang")
_CBENCH_DATA = runfiles_path("CompilerGym/compiler_gym/third_party/cBench/runtime_data")

if sys.platform == "darwin":
    _COMPILE_ARGS = [
        "-L",
        "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib",
    ]
else:
    _COMPILE_ARGS = []

LLVM_DATASETS = [
    Dataset(
        name="blas-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-blas-v0.tar.bz2",
        license="BSD 3-Clause",
        description="https://github.com/spcl/ncc/tree/master/data",
        compiler="llvm-10.0.0",
        file_count=300,
        size_bytes=3969036,
        sha256="e724a8114709f8480adeb9873d48e426e8d9444b00cddce48e342b9f0f2b096d",
    ),
    Dataset(
        name="cBench-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-cBench-v0-macos.tar.bz2",
        license="BSD 3-Clause",
        description="https://github.com/ctuning/ctuning-programs",
        compiler="llvm-10.0.0",
        file_count=23,
        size_bytes=7150112,
        sha256="68e4ead67ded0ea8772f9d6796579115dd7ea01f6ad2132a127f034549e64940",
        platforms=["macos"],
    ),
    Dataset(
        name="cBench-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-cBench-v0-linux.tar.bz2",
        license="BSD 3-Clause",
        description="https://github.com/ctuning/ctuning-programs",
        compiler="llvm-10.0.0",
        file_count=23,
        size_bytes=7150112,
        sha256="9b5838a90895579aab3b9375e8eeb3ed2ae58e0ad354fec7eb4f8b31ecb4a360",
        platforms=["linux"],
    ),
    Dataset(
        name="github-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-github-v0.tar.bz2",
        license="CC BY 4.0",
        description="https://zenodo.org/record/4122437",
        compiler="llvm-10.0.0",
        file_count=50708,
        size_bytes=725974100,
        sha256="880269dd7a5c2508ea222a2e54c318c38c8090eb105c0a87c595e9dd31720764",
    ),
    Dataset(
        name="linux-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-linux-v0.tar.bz2",
        license="GPL-2.0",
        description="https://github.com/spcl/ncc/tree/master/data",
        compiler="llvm-10.0.0",
        file_count=13920,
        size_bytes=516031044,
        sha256="a1ae5c376af30ab042c9e54dc432f89ce75f9ebaee953bc19c08aff070f12566",
    ),
    Dataset(
        name="mibench-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-mibench-v0.tar.bz2",
        license="BSD 3-Clause",
        description="https://github.com/ctuning/ctuning-programs",
        compiler="llvm-10.0.0",
        file_count=40,
        size_bytes=238480,
        sha256="128c090c40b955b99fdf766da167a5f642018fb35c16a1d082f63be2e977eb13",
    ),
    Dataset(
        name="npb-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-npb-v0.tar.bz2",
        license="NASA Open Source Agreement v1.3",
        description="https://github.com/spcl/ncc/tree/master/data",
        compiler="llvm-10.0.0",
        file_count=122,
        size_bytes=2287444,
        sha256="793ac2e7a4f4ed83709e8a270371e65b724da09eaa0095c52e7f4209f63bb1f2",
    ),
    Dataset(
        name="opencv-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-opencv-v0.tar.bz2",
        license="Apache 2.0",
        description="https://github.com/spcl/ncc/tree/master/data",
        compiler="llvm-10.0.0",
        file_count=442,
        size_bytes=21903008,
        sha256="003df853bd58df93572862ca2f934c7b129db2a3573bcae69a2e59431037205c",
    ),
    Dataset(
        name="poj104-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-poj104-v0.tar.bz2",
        license="BSD 3-Clause",
        description="https://sites.google.com/site/treebasedcnn/",
        compiler="llvm-10.0.0",
        file_count=49628,
        size_bytes=304207752,
        sha256="6254d629887f6b51efc1177788b0ce37339d5f3456fb8784415ed3b8c25cce27",
    ),
    Dataset(
        name="polybench-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-polybench-v0.tar.bz2",
        license="BSD 3-Clause",
        description="https://github.com/ctuning/ctuning-programs",
        compiler="llvm-10.0.0",
        file_count=27,
        size_bytes=162624,
        sha256="968087e68470e5b44dc687dae195143000c7478a23d6631b27055bb3bb3116b1",
    ),
    Dataset(
        name="tensorflow-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-tensorflow-v0.tar.bz2",
        license="Apache 2.0",
        description="https://github.com/spcl/ncc/tree/master/data",
        compiler="llvm-10.0.0",
        file_count=1985,
        size_bytes=299697312,
        sha256="f77dd1988c772e8359e1303cc9aba0d73d5eb27e0c98415ac3348076ab94efd1",
    ),
]


class BenchmarkExecutionResult(NamedTuple):
    """The result of running a benchmark."""

    walltime_seconds: float
    """The execution time in seconds."""

    error: Optional[str] = None
    """An error message."""

    output: Optional[str] = None
    """The output generated by the benchmark."""


def _run_cbench_benchmark(
    env: "LlvmEnv",
    cmd: str,
    linkopts: Tuple[str],
    num_runs: int,
    ignore_returncode: bool,
    timeout_seconds: float = 60,
) -> BenchmarkExecutionResult:
    """Run the given cBench benchmark."""
    data = _CBENCH_DATA / env.benchmark.split("/")[-1]

    # cBench benchmarks expect that a file _finfo_dataset exists in the
    # current working directory and contains the number of benchmark
    # iterations in it.
    with open("_finfo_dataset", "w") as f:
        print(num_runs, file=f)

    # Serialize the benchmark to a bitcode file that will then be
    # compiled to a binary.
    bitcode_file = Path(env.observation["BitcodeFile"])
    try:
        # Generate the a.out binary file.
        subprocess.check_call(
            [_CLANG, str(bitcode_file)] + _COMPILE_ARGS + list(linkopts)
        )
        assert Path("a.out").is_file()

        cmd = expand_command_vars(cmd)
        process = subprocess.Popen(
            cmd,
            shell=True,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            env=os.environ,
        )

        try:
            with Timer() as timer:
                stdout, _ = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            return BenchmarkExecutionResult(
                walltime_seconds=timeout_seconds,
                error=f"Benchmark failed to complete with {timeout_seconds} timeout.",
            )

        if not ignore_returncode and process.returncode:
            output = None
            try:
                output = stdout.decode("utf-8")
            except UnicodeDecodeError:
                pass
            return BenchmarkExecutionResult(
                walltime_seconds=timer.time,
                error=f"Benchmark exited with returncode {process.returncode}. Output: {output}",
            )
        return BenchmarkExecutionResult(walltime_seconds=timer.time, output=stdout)
    finally:
        bitcode_file.unlink()


@lru_cache(maxsize=1024)
def get_cBench_reference_output(
    benchmark: str, cmd: str, linkopts: Tuple[str], **hashopts
) -> str:
    """Produce a gold-standard benchmark output."""
    env = gym.make("llvm-v0")
    try:
        env.reset(benchmark=benchmark)
        outcome = _run_cbench_benchmark(
            env, cmd, num_runs=1, ignore_returncode=False, linkopts=linkopts
        )
        if outcome.error:
            raise OSError(
                f"Failed to produce reference output for benchmark '{env.benchmark}' using '{cmd}': {outcome.error}"
            )
    finally:
        env.close()
    return outcome.output


def make_cBench_validator(
    cmd: str,
    linkopts: Tuple[str],
    os_env: Dict[str, str],
    num_runs: int = 1,
    ignore_returncode: bool = False,
    compare_output: bool = True,
    output_files: Optional[List[Path]] = None,
    validate_result: Optional[
        Callable[[BenchmarkExecutionResult], Optional[str]]
    ] = None,
    pre_execution_callback: Optional[Callable[[], None]] = None,
):
    """Generate a validation callback for a cBench benchmark."""
    output_files = output_files or []

    def validator(env):
        """The validation callback."""
        with temporary_working_directory():
            with benchmark_execution_environment(os_env):
                if pre_execution_callback:
                    pre_execution_callback()

                # Produce a gold-standard output using a reference version of
                # the benchmark.
                if compare_output or output_files:
                    gold_standard = get_cBench_reference_output(
                        env.benchmark,
                        cmd,
                        linkopts=linkopts,
                        os_env_hash=",".join(f"{k}:{v}" for k, v in os_env.items()),
                    )

                    for path in output_files:
                        print(list(Path(".").iterdir()))
                        if not path.is_file():
                            raise FileNotFoundError(
                                f"Expected file not generated by benchmark {env.benchmark}: {path}.\nCommand: {cmd}"
                            )
                        path.rename(f"{path}.gold_standard")

                outcome = _run_cbench_benchmark(
                    env,
                    cmd,
                    num_runs=num_runs,
                    ignore_returncode=ignore_returncode,
                    linkopts=linkopts,
                )

            if outcome.error:
                return outcome.error

            # Run a user-specified validation hook.
            if validate_result:
                validate_result(outcome)

            # Difftest the console output.
            if compare_output and gold_standard != outcome.output:
                return (
                    f"Benchmark output differs from expected.\n"
                    f"Expected: {gold_standard}\n"
                    f"Actual: {outcome.output}"
                )

            # Difftest the output files.
            for path in output_files:
                if not path.is_file():
                    return f"Expected file not generated by benchmark {env.benchmark}: {path}.\nCommand: {cmd}"
                diff = subprocess.Popen(["diff", str(path), f"{path}.gold_standard"])
                diff.communicate()
                if diff.returncode:
                    return (
                        "Benchmark produced output file with invalid contents: {path}"
                    )

    return validator


@contextmanager
def temporary_environment():
    """Yield a temporary os.environ state."""
    _environ = os.environ.copy()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(_environ)


@contextmanager
def benchmark_execution_environment(env: Dict[str, str]):
    """Setup the os.environ state for executing benchmarks."""
    with temporary_environment():
        for var in os.environ:
            if not (
                var.startswith("COMPILER_GYM_")
                or var in {"PATH", "RUNFILES_DIR", "RUNFILES_MANIFEST_FILE"}
            ):
                del os.environ[var]
        for key, val in env.items():
            os.environ[key] = expand_command_vars(val)
        yield


def expand_command_vars(cmd: str) -> str:
    with temporary_environment():
        os.environ.clear()
        os.environ["BIN"] = "a.out"
        os.environ["D"] = str(_CBENCH_DATA)
        os.environ["O"] = str(_CBENCH_DATA)
        return os.path.expandvars(cmd)


_VALIDATORS: Dict[str, List[Callable[["LlvmEnv"], Optional[str]]]] = defaultdict(list)


def validator(
    benchmark: str,
    cmd: str,
    data: Optional[List[str]] = None,
    outs: Optional[List[str]] = None,
    platforms: Optional[List[str]] = None,
    compare_output: bool = True,
    validate_result: Optional[
        Callable[[BenchmarkExecutionResult], Optional[str]]
    ] = None,
    linkopts: List[str] = None,
    env: Dict[str, str] = None,
    pre_execution_callback: Optional[Callable[[], None]] = None,
) -> bool:
    """Declare a new benchmark validator.

    :param benchmark: The name of the benchmark that this validator supports.
    :cmd: The shell command to run the validation. Variable substitution is applied to this value as follows: :code:`$BIN` is replaced by the path of the compiled binary and :code:`$D` is replaced with the path to the benchmark's runtime data directory.
    :data: A list of paths to input files.
    :outs: A list of paths to output files.
    :return: :code:`True` if the new validator was registered, else :code:`False`.
    """
    platforms = platforms or ["linux", "macos"]
    if {"darwin": "macos"}.get(sys.platform, sys.platform) not in platforms:
        return False
    infiles = [_CBENCH_DATA / p for p in data or []]
    outfiles = [Path(p) for p in outs or []]
    linkopts = tuple(linkopts or [])
    env = env or {}

    for path in infiles:
        if not path.is_file():
            raise FileNotFoundError(f"Required benchmark input not found: {path}")

    _VALIDATORS[benchmark].append(
        make_cBench_validator(
            cmd=cmd,
            output_files=outfiles,
            compare_output=compare_output,
            validate_result=validate_result,
            linkopts=linkopts,
            os_env=env,
            pre_execution_callback=pre_execution_callback,
        )
    )
    return True


def _compose_validators(*validators):
    def composed(env):
        # Shortcut when there is only a single validator to run.
        if len(validators):
            return validators[0](env)

        # Run validators simultaneously in parallel threads.
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(validator, env) for validator in validators]
            result = None
            for future in as_completed(futures):
                result = future.result() or result
            return result

    return composed


def get_llvm_benchmark_validation_callback(
    env: "LlvmEnv",
) -> Optional[Callable[["LlvmEnv"], Optional[str]]]:
    """"""
    val = _VALIDATORS.get(env.benchmark)
    return _compose_validators(*val) if val else None


# ===============================
# Definition of cBench validators
# ===============================


def validate_sha_output(result: BenchmarkExecutionResult):
    # SHA benchmark prints 5 random hex strings. Normally these hex strings are
    # 16 characters but occasionally they are less (presumably becuase of a
    # leading zero being omitted).
    assert re.match(
        r"[0-9a-f]{0,16} [0-9a-f]{0,16} [0-9a-f]{0,16} [0-9a-f]{0,16} [0-9a-f]{0,16}",
        result.output.decode("utf-8").rstrip(),
    )


def setup_ghostscript_library_files():
    # Copy input data into current directory since ghostscript doesn't like long
    # input paths.
    for path in (_CBENCH_DATA / "office_data").iterdir():
        if path.name.endswith(".ps"):
            shutil.copyfile(path, path.name)
    # Ghostscript doesn't like the library files being symlinks so copy them
    # into the working directory as regular files.
    for path in (_CBENCH_DATA / "ghostscript").iterdir():
        if path.name.endswith(".ps"):
            shutil.copyfile(path, path.name)


validator(
    benchmark="benchmark://cBench-v0/bitcount",
    cmd="$BIN 1125000",
)

validator(
    benchmark="benchmark://cBench-v0/bitcount",
    cmd="$BIN 512",
)

for i in range(1, 21):
    validator(
        benchmark="benchmark://cBench-v0/adpcm",
        cmd=f"$BIN $D/telecom_data/{i}.adpcm",
        data=[f"telecom_data/{i}.adpcm"],
    )

    validator(
        benchmark="benchmark://cBench-v0/adpcm",
        cmd=f"$BIN $D/telecom_data/{i}.pcm",
        data=[f"telecom_data/{i}.pcm"],
    )

    validator(
        benchmark="benchmark://cBench-v0/blowfish",
        cmd=f"$BIN d $D/office_data/{i}.benc output.txt 1234567890abcdeffedcba0987654321",
        data=[f"office_data/{i}.benc"],
        outs=["output.txt"],
    )

    validator(
        benchmark="benchmark://cBench-v0/bzip2",
        cmd=f"$BIN -d -k -f -c $D/bzip2_data/{i}.bz2",
        data=[f"bzip2_data/{i}.bz2"],
    )

    validator(
        benchmark="benchmark://cBench-v0/crc32",
        cmd=f"$BIN $D/telecom_data/{i}.pcm",
        data=[f"telecom_data/{i}.pcm"],
    )

    validator(
        benchmark="benchmark://cBench-v0/dijkstra",
        cmd=f"$BIN $D/network_dijkstra_data/{i}.dat",
        data=[f"network_dijkstra_data/{i}.dat"],
    )

    validator(
        benchmark="benchmark://cBench-v0/ghostscript",
        cmd=f"$BIN -sDEVICE=ppm -dNOPAUSE -dQUIET -sOutputFile=output.ppm -- {i}.ps",
        data=[f"office_data/{i}.ps"],
        outs=["output.ppm"],
        linkopts=["-lm", "-lz"],
        pre_execution_callback=setup_ghostscript_library_files,
    )

    validator(
        benchmark="benchmark://cBench-v0/gsm",
        cmd=f"$BIN -fps -c $D/telecom_gsm_data/{i}.au",
        data=[f"telecom_gsm_data/{i}.au"],
    )

    # TODO(cummins): ispell executable appears broken. Investigation needed.
    # validator(
    #     benchmark="benchmark://cBench-v0/ispell",
    #     cmd=f"$BIN -a -d americanmed+ $D/office_data/{i}.txt",
    #     data = [f"office_data/{i}.txt"],
    # )

    validator(
        benchmark="benchmark://cBench-v0/jpeg-c",
        cmd=f"$BIN -dct int -progressive -outfile output.jpeg $D/consumer_jpeg_data/{i}.ppm",
        data=[f"consumer_jpeg_data/{i}.ppm"],
        outs=["output.jpeg"],
    )

    validator(
        benchmark="benchmark://cBench-v0/jpeg-d",
        cmd=f"$BIN -dct int -outfile output.ppm $D/consumer_jpeg_data/{i}.jpg",
        data=[f"consumer_jpeg_data/{i}.jpg"],
        outs=["output.ppm"],
    )
    validator(
        benchmark="benchmark://cBench-v0/lame",
        cmd=f"$BIN $D/consumer_data/{i}.wav output.mp3",
        data=[f"consumer_data/{i}.wav"],
        outs=["output.mp3"],
        compare_output=False,
        linkopts=["-lm"],
    )

    validator(
        benchmark="benchmark://cBench-v0/patricia",
        cmd=f"$BIN $D/network_patricia_data/{i}.udp",
        data=[f"network_patricia_data/{i}.udp"],
    )

    validator(
        benchmark="benchmark://cBench-v0/qsort",
        cmd=f"$BIN $D/automotive_qsort_data/{i}.dat",
        data=[f"automotive_qsort_data/{i}.dat"],
        outs=["sorted_output.dat"],
        linkopts=["-lm"],
    )

    validator(
        benchmark="benchmark://cBench-v0/rijndael",
        cmd=f"$BIN $D/office_data/{i}.enc output.dec d 1234567890abcdeffedcba09876543211234567890abcdeffedcba0987654321",
        data=[f"office_data/{i}.enc"],
        outs=["output.dec"],
    )

    validator(
        benchmark="benchmark://cBench-v0/sha",
        cmd=f"$BIN $D/office_data/{i}.txt",
        data=[f"office_data/{i}.txt"],
        compare_output=False,
        validate_result=validate_sha_output,
    )

    validator(
        benchmark="benchmark://cBench-v0/stringsearch",
        cmd=f"$BIN $D/office_data/{i}.txt $D/office_data/{i}.s.txt output.txt",
        data=[f"office_data/{i}.txt"],
        outs=["output.txt"],
        linkopts=["-lm"],
    )

    validator(
        benchmark="benchmark://cBench-v0/stringsearch2",
        cmd=f"$BIN $D/office_data/{i}.txt $D/office_data/{i}.s.txt output.txt",
        data=[f"office_data/{i}.txt"],
        outs=["output.txt"],
    )

    validator(
        benchmark="benchmark://cBench-v0/susan",
        cmd=f"$BIN $D/automotive_susan_data/{i}.pgm output_large.corners.pgm -c",
        data=[f"automotive_susan_data/{i}.pgm"],
        outs=["output_large.corners.pgm"],
        linkopts=["-lm"],
    )

    validator(
        benchmark="benchmark://cBench-v0/tiff2bw",
        cmd=f"$BIN $D/consumer_tiff_data/{i}.tif output.tif",
        data=[f"consumer_tiff_data/{i}.tif"],
        outs=["output.tif"],
        linkopts=["-lm"],
    )

    validator(
        benchmark="benchmark://cBench-v0/tiff2rgba",
        cmd=f"$BIN $D/consumer_tiff_data/{i}.tif output.tif",
        data=[f"consumer_tiff_data/{i}.tif"],
        outs=["output.tif"],
        linkopts=["-lm"],
    )

    validator(
        benchmark="benchmark://cBench-v0/tiffdither",
        cmd=f"$BIN $D/consumer_tiff_data/{i}.bw.tif out.tif",
        data=[f"consumer_tiff_data/{i}.bw.tif"],
        outs=["out.tif"],
        linkopts=["-lm"],
    )

    validator(
        benchmark="benchmark://cBench-v0/tiffmedian",
        cmd=f"$BIN $D/consumer_tiff_data/{i}.nocomp.tif output.tif",
        data=[f"consumer_tiff_data/{i}.nocomp.tif"],
        outs=["output.tif"],
        linkopts=["-lm"],
    )
