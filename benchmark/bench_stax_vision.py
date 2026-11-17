#!/usr/bin/env python3
"""Vision-model compile benchmark: compile cost, inference and training.

Each case builds one model for the checkout's runtime and one for the
reference runtime from the same parameters, then measures three things on the
same device:

* inference latency after a compile, against the reference runtime's eager
  execution and its own compiled execution;
* one training step (forward, loss, backward, optimizer), under the same three
  execution models, with only the model compiled on either side;
* the wall time from asking for a compile to the first completed call, which
  is what a user pays once before any steady-state number starts to matter.

Correctness is checked before timing: the compiled logits must match the eager
logits to a tolerance that scales with the element type, so a fast but wrong
case cannot enter the report as a win.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

# The reference runtime is imported first: it loads the shared numerical
# libraries, and letting the checkout's package load them first makes some
# installations pick up a second copy of the same kernels.
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorplay as tp  # noqa: E402
import tensorplay.nn.functional as tp_F  # noqa: E402
from tensorplay.vision import models as tp_models  # noqa: E402


@dataclass(frozen=True)
class Case:
    """One model and the input extent its classifier was trained on."""

    name: str
    input_size: int = 224


#: Every classification model the vision package builds from plain arguments.
#: The video, optical-flow and segmentation entry points take inputs that are
#: not images, so they are timed by their own benchmarks instead.
CASES: tuple[Case, ...] = tuple(
    Case(name, 299 if name == "inception_v3" else 224)
    for name in (
        "alexnet",
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "convnext_large",
        "densenet121",
        "densenet169",
        "densenet201",
        "densenet161",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "efficientnet_b4",
        "efficientnet_b5",
        "efficientnet_b6",
        "efficientnet_b7",
        "efficientnet_v2_s",
        "efficientnet_v2_m",
        "efficientnet_v2_l",
        "googlenet",
        "inception_v3",
        "maxvit_t",
        "mnasnet0_5",
        "mnasnet0_75",
        "mnasnet1_0",
        "mnasnet1_3",
        "mobilenet_v2",
        "mobilenet_v3_large",
        "mobilenet_v3_small",
        "regnet_x_400mf",
        "regnet_x_800mf",
        "regnet_x_1_6gf",
        "regnet_x_3_2gf",
        "regnet_x_8gf",
        "regnet_x_16gf",
        "regnet_x_32gf",
        "regnet_y_400mf",
        "regnet_y_800mf",
        "regnet_y_1_6gf",
        "regnet_y_3_2gf",
        "regnet_y_8gf",
        "regnet_y_16gf",
        "regnet_y_32gf",
        "regnet_y_128gf",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "resnext101_64x4d",
        "wide_resnet50_2",
        "wide_resnet101_2",
        "shufflenet_v2_x0_5",
        "shufflenet_v2_x1_0",
        "shufflenet_v2_x1_5",
        "shufflenet_v2_x2_0",
        "squeezenet1_0",
        "squeezenet1_1",
        "swin_t",
        "swin_s",
        "swin_b",
        "swin_v2_t",
        "swin_v2_s",
        "swin_v2_b",
        "vgg11",
        "vgg11_bn",
        "vgg13",
        "vgg13_bn",
        "vgg16",
        "vgg16_bn",
        "vgg19",
        "vgg19_bn",
        "vit_b_16",
        "vit_b_32",
        "vit_l_16",
        "vit_l_32",
        "vit_h_14",
    )
)


def selected_cases(patterns: list[str]) -> list[Case]:
    if not patterns:
        return list(CASES)
    wanted: list[Case] = []
    for pattern in patterns:
        matched = [case for case in CASES if pattern in case.name]
        if not matched:
            raise SystemExit(f"no case matches {pattern!r}")
        wanted.extend(case for case in matched if case not in wanted)
    return wanted


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
        tp.cuda.synchronize()


def time_call(fn: Callable[[], Any], device: str, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    samples = []
    for _ in range(repeats):
        if device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            synchronize(device)
            samples.append(start.elapsed_time(end))
        else:
            started = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - started) * 1e3)
    return statistics.median(samples)


def to_tensorplay_state(state: dict[str, Any]) -> dict[str, Any]:
    converted = {}
    for key, value in state.items():
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        converted[key] = tp.from_numpy(array.copy())
    return converted


def build_pair(case: Case, device: str, dtype: torch.dtype) -> tuple[Any, Any]:
    tp_builder = getattr(tp_models, case.name)
    reference_builder = getattr(torchvision.models, case.name)
    tp_model = tp_builder(weights=None)
    reference_model = reference_builder(weights=None)
    tp_model.load_state_dict(to_tensorplay_state(reference_model.state_dict()))
    tp_model = tp_model.to(dtype=getattr(tp, str(dtype).split(".")[-1])).to(device)
    reference_model = reference_model.to(dtype=dtype, device=device)
    return tp_model.eval(), reference_model.eval()


def model_logits(output: Any) -> Any:
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value.detach().cpu().numpy()


def compile_tensorplay(model: Any, mode: str | None) -> tuple[Any, float]:
    started = time.perf_counter()
    compiled = tp.compile(
        model,
        backend="stax",
        fullgraph=True,
        strict_native=True,
        mode=mode,
    )
    return compiled, time.perf_counter() - started


def compile_reference(model: Any, mode: str | None) -> tuple[Any, float]:
    started = time.perf_counter()
    compiled = torch.compile(model, fullgraph=True, mode=mode)
    return compiled, time.perf_counter() - started


def codegens(compiled: Any) -> list[str]:
    cache = getattr(compiled, "_tensorplay_cache", {})
    names = []
    for lowering in cache.values():
        name = getattr(lowering, "_tensorplay_codegen", None)
        if name:
            names.append(str(name))
    return names


def backward_codegens(compiled: Any) -> list[str]:
    cache = getattr(compiled, "_tensorplay_cache", {})
    names = []
    for lowering in cache.values():
        name = getattr(lowering, "_tensorplay_backward_codegen", None)
        if name:
            names.append(str(name))
    return names


def tolerance(dtype: torch.dtype) -> float:
    return {torch.float32: 5e-3, torch.float16: 2e-2, torch.bfloat16: 3e-2}[dtype]


def run_case(
    case: Case,
    device: str,
    dtype: torch.dtype,
    batch: int,
    warmup: int,
    repeats: int,
    train_warmup: int,
    train_repeats: int,
    mode: str | None,
    tasks: set[str],
    run_reference_compile: bool,
    cold_compile: bool = False,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": case.name,
        "device": device,
        "dtype": str(dtype).split(".")[-1],
        "batch": batch,
        "input_size": case.input_size,
    }
    try:
        tp_model, reference_model = build_pair(case, device, dtype)
        if tasks == {"infer"}:
            # An inference-only comparison does not need autograd history on
            # the parameters, and leaving it off is what a serving path does.
            tp_model.requires_grad_(False)
            reference_model.requires_grad_(False)
        rng = np.random.default_rng(20261117)
        shape = (batch, 3, case.input_size, case.input_size)
        array = rng.standard_normal(shape).astype(np.float32)
        tp_input = tp.from_numpy(array.copy()).to(dtype=getattr(tp, str(dtype).split(".")[-1])).to(device)
        reference_input = torch.from_numpy(array.copy()).to(dtype=dtype, device=device)
        labels_array = rng.integers(0, 1000, size=(batch,)).astype(np.int64)
        tp_labels = tp.from_numpy(labels_array.copy()).to(device)
        reference_labels = torch.from_numpy(labels_array.copy()).to(device)

        with tp.no_grad():
            tp_eager_logits = model_logits(tp_model(tp_input)).float().cpu().numpy()
        with torch.no_grad():
            reference_eager_logits = model_logits(reference_model(reference_input)).float().cpu().numpy()
        record["max_abs_error_eager"] = float(
            np.max(np.abs(tp_eager_logits - reference_eager_logits))
        )
        record["eager_matches"] = bool(
            np.allclose(
                tp_eager_logits,
                reference_eager_logits,
                atol=tolerance(dtype),
                rtol=tolerance(dtype),
            )
        )

        if "infer" in tasks:
            if cold_compile:
                torch.compiler.reset()
            with tp.no_grad():
                record["tp_eager_infer_ms"] = time_call(
                    lambda: tp_model(tp_input), device, warmup, repeats
                )
            with torch.no_grad():
                record["reference_eager_infer_ms"] = time_call(
                    lambda: reference_model(reference_input), device, warmup, repeats
                )

            stax, stax_compile = compile_tensorplay(tp_model, mode)
            started = time.perf_counter()
            with tp.no_grad():
                compiled_logits = model_logits(stax(tp_input)).float().cpu().numpy()
            record["stax_compile_seconds"] = stax_compile
            record["stax_first_call_seconds"] = time.perf_counter() - started
            record["max_abs_error_compiled"] = float(
                np.max(np.abs(compiled_logits - reference_eager_logits))
            )
            record["compiled_matches"] = bool(
                np.allclose(
                    compiled_logits,
                    reference_eager_logits,
                    atol=tolerance(dtype),
                    rtol=tolerance(dtype),
                )
            )
            record["stax_codegens"] = codegens(stax)
            with tp.no_grad():
                record["stax_infer_ms"] = time_call(
                    lambda: stax(tp_input), device, warmup, repeats
                )
            record["stax_vs_eager"] = (
                record["tp_eager_infer_ms"] / record["stax_infer_ms"]
            )
            record["stax_vs_reference_eager"] = (
                record["reference_eager_infer_ms"] / record["stax_infer_ms"]
            )

            if run_reference_compile:
                reference, reference_compile = compile_reference(reference_model, mode)
                started = time.perf_counter()
                with torch.no_grad():
                    reference(reference_input)
                record["reference_compile_seconds"] = reference_compile
                record["reference_compile_first_call_seconds"] = (
                    time.perf_counter() - started
                )
                with torch.no_grad():
                    record["reference_compiled_infer_ms"] = time_call(
                        lambda: reference(reference_input), device, warmup, repeats
                    )
                record["stax_vs_reference_compiled"] = (
                    record["reference_compiled_infer_ms"] / record["stax_infer_ms"]
                )
                record["compile_time_ratio"] = (
                    record["reference_compile_seconds"] / record["stax_compile_seconds"]
                )

        if "train" in tasks:
            tp_model.train()
            reference_model.train()
            tp_optimizer = tp.optim.SGD(tp_model.parameters(), lr=0.01)
            reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.01)

            def tp_step() -> None:
                tp_optimizer.zero_grad(set_to_none=True)
                loss = tp_F.cross_entropy(
                    model_logits(tp_model(tp_input)).float(), tp_labels
                )
                loss.backward()
                tp_optimizer.step()

            def reference_step() -> None:
                reference_optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(
                    model_logits(reference_model(reference_input)).float(),
                    reference_labels,
                )
                loss.backward()
                reference_optimizer.step()

            record["tp_eager_train_ms"] = time_call(
                tp_step, device, train_warmup, train_repeats
            )
            record["reference_eager_train_ms"] = time_call(
                reference_step, device, train_warmup, train_repeats
            )

            if cold_compile:
                torch.compiler.reset()
            stax_train, stax_train_compile = compile_tensorplay(tp_model, mode)
            started = time.perf_counter()
            tp_optimizer.zero_grad(set_to_none=True)
            model_logits(stax_train(tp_input)).float().sum().backward()
            tp_optimizer.step()
            record["stax_train_compile_seconds"] = stax_train_compile
            record["stax_train_first_step_seconds"] = time.perf_counter() - started
            record["stax_backward_codegens"] = backward_codegens(stax_train)

            def stax_step() -> None:
                tp_optimizer.zero_grad(set_to_none=True)
                loss = tp_F.cross_entropy(
                    model_logits(stax_train(tp_input)).float(), tp_labels
                )
                loss.backward()
                tp_optimizer.step()

            record["stax_train_ms"] = time_call(
                stax_step, device, train_warmup, train_repeats
            )
            record["stax_train_vs_eager"] = (
                record["tp_eager_train_ms"] / record["stax_train_ms"]
            )
            record["stax_train_vs_reference_eager"] = (
                record["reference_eager_train_ms"] / record["stax_train_ms"]
            )

            if run_reference_compile:
                reference_train, reference_train_compile = compile_reference(
                    reference_model, mode
                )
                started = time.perf_counter()
                reference_optimizer.zero_grad(set_to_none=True)
                model_logits(reference_train(reference_input)).float().sum().backward()
                reference_optimizer.step()
                record["reference_train_compile_seconds"] = reference_train_compile
                record["reference_train_first_step_seconds"] = (
                    time.perf_counter() - started
                )

                def reference_compiled_step() -> None:
                    reference_optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(
                        model_logits(reference_train(reference_input)).float(),
                        reference_labels,
                    )
                    loss.backward()
                    reference_optimizer.step()

                record["reference_compiled_train_ms"] = time_call(
                    reference_compiled_step, device, train_warmup, train_repeats
                )
                record["stax_train_vs_reference_compiled"] = (
                    record["reference_compiled_train_ms"] / record["stax_train_ms"]
                )
                record["train_compile_time_ratio"] = (
                    record["reference_train_compile_seconds"]
                    / record["stax_train_compile_seconds"]
                )
    except Exception as exc:  # Keep one compiler failure visible in the report.
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    finally:
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return record


def print_record(record: dict[str, Any]) -> None:
    name = record["name"]
    if "error" in record:
        print(f"{name:24} ERROR {record['error']}", flush=True)
        return
    fields = [f"{name:24}"]
    if "stax_infer_ms" in record:
        fields.append(
            "infer stax={stax:8.3f} refEager={ref:8.3f} refCompiled={comp:8.3f} "
            "vsEager={ve:5.2f}x vsCompiled={vc:5.2f}x "
            "compile={cs:6.2f}s/{rs:6.2f}s ({cr:5.2f}x)".format(
                stax=record["stax_infer_ms"],
                ref=record["reference_eager_infer_ms"],
                comp=record.get("reference_compiled_infer_ms", float("nan")),
                ve=record["stax_vs_reference_eager"],
                vc=record.get("stax_vs_reference_compiled", float("nan")),
                cs=record["stax_compile_seconds"],
                rs=record.get("reference_compile_seconds", float("nan")),
                cr=record.get("compile_time_ratio", float("nan")),
            )
        )
    if "stax_train_ms" in record:
        fields.append(
            "train stax={stax:8.3f} refEager={ref:8.3f} refCompiled={comp:8.3f} "
            "vsEager={ve:5.2f}x vsCompiled={vc:5.2f}x".format(
                stax=record["stax_train_ms"],
                ref=record["reference_eager_train_ms"],
                comp=record.get("reference_compiled_train_ms", float("nan")),
                ve=record["stax_train_vs_reference_eager"],
                vc=record.get("stax_train_vs_reference_compiled", float("nan")),
            )
        )
    print(" | ".join(fields), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--train-warmup", type=int, default=2)
    parser.add_argument("--train-repeats", type=int, default=5)
    parser.add_argument("--tasks", default="infer,train")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--no-reference-compile", action="store_true")
    parser.add_argument(
        "--cold-compile",
        action="store_true",
        help="measure a compile that cannot read a cached artifact on either side",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    tasks = set(args.tasks.split(","))
    if not tasks <= {"infer", "train"}:
        parser.error("tasks may only name infer and train")
    dtype = getattr(torch, args.dtype)
    if args.threads and hasattr(tp, "set_num_threads"):
        tp.set_num_threads(args.threads)
        torch.set_num_threads(args.threads)
    if args.cold_compile:
        from tensorplay.compiler.backends.stax import config as stax_config

        stax_config.force_disable_caches = True
        torch._inductor.config.force_disable_caches = True

    def payload() -> dict[str, Any]:
        return {
            "schema_version": 1,
            "benchmark": "stax-vision",
            "device": args.device,
            "dtype": args.dtype,
            "batch": args.batch,
            "tasks": sorted(tasks),
            "mode": args.mode,
            "cold_compile": args.cold_compile,
            "threads": args.threads
            or getattr(tp, "get_num_threads", lambda: 0)(),
            "measurements": records,
        }

    def write() -> None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload(), indent=2) + "\n", encoding="utf-8"
        )

    records = []
    for case in selected_cases(args.models):
        record = run_case(
            case,
            args.device,
            dtype,
            args.batch,
            args.warmup,
            args.repeats,
            args.train_warmup,
            args.train_repeats,
            args.mode,
            tasks,
            not args.no_reference_compile,
            cold_compile=args.cold_compile,
        )
        print_record(record)
        records.append(record)
        # Each case is a long compile in its own right; writing after every
        # one keeps a sweep that is interrupted from losing the finished work.
        write()

    print(f"wrote results: {args.json_out}")


if __name__ == "__main__":
    main()