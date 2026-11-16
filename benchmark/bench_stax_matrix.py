"""Compare compiled steady-state execution across representative graphs."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import tensorplay as tp


CASES = {
    "pointwise": (((1024, 1024),), lambda m, x: (x.sin() + x.cos()) * 0.5 + x.relu()),
    "reduce_rows": (((1024, 1024),), lambda m, x: x.sum(dim=1)),
    "reduce_columns": (((1024, 1024),), lambda m, x: x.sum(dim=0)),
    "softmax": (((512, 1024),), lambda m, x: m.softmax(x, dim=-1)),
    "softmax_small": (((32, 128),), lambda m, x: m.softmax(x, dim=-1)),
    "softmax_wide": (((128, 4096),), lambda m, x: m.softmax(x, dim=-1)),
    "softmax_axis": (((256, 512),), lambda m, x: m.softmax(x, dim=0)),
    "layer_norm": (((512, 1024),), lambda m, x: m.nn.functional.layer_norm(x, (1024,))),
    "layer_norm_small": (((32, 128),), lambda m, x: m.nn.functional.layer_norm(x, (128,))),
    "layer_norm_wide": (((128, 4096),), lambda m, x: m.nn.functional.layer_norm(x, (4096,))),
    "matmul": (((256, 512), (512, 512)), lambda m, x, y: x @ y),
    "linear_relu": (((256, 512), (512, 512), (512,)), lambda m, x, w, b: (x @ w + b).relu()),
    "conv_relu": (((8, 32, 32, 32), (64, 32, 3, 3)), lambda m, x, w: m.nn.functional.conv2d(x, w, padding=1).relu()),
}


def measure(device, warmup, iterations, rounds, selected):
    tp.set_num_threads(1)
    torch.set_num_threads(1)
    rng = np.random.default_rng(20261116)
    results = []

    def sync():
        if device == "cuda":
            tp.cuda.synchronize()
            torch.cuda.synchronize()

    for name, (shapes, operation) in CASES.items():
        if selected and name not in selected:
            continue
        arrays = [rng.standard_normal(shape, dtype=np.float32) for shape in shapes]
        tp_args = [tp.from_numpy(a.copy()).to(device) for a in arrays]
        reference_args = [torch.from_numpy(a.copy()).to(device) for a in arrays]

        if len(shapes) == 1:
            def tp_region(x):
                return operation(tp, x)

            def reference_region(x):
                return operation(torch, x)
        elif len(shapes) == 2:
            def tp_region(x, y):
                return operation(tp, x, y)

            def reference_region(x, y):
                return operation(torch, x, y)
        else:
            def tp_region(x, y, z):
                return operation(tp, x, y, z)

            def reference_region(x, y, z):
                return operation(torch, x, y, z)

        try:
            torch._dynamo.reset()
            compiled = tp.compile(tp_region, strict_native=True, fullgraph=True)
            reference = torch.compile(reference_region, fullgraph=True)
            result = compiled(*tp_args)
            expected = reference(*reference_args)
            sync()
            np.testing.assert_allclose(result.cpu().numpy(), expected.cpu().numpy(), rtol=2e-3, atol=2e-3)
            calls = [lambda: compiled(*tp_args), lambda: reference(*reference_args)]
            for _ in range(warmup):
                for call in calls:
                    call()
            sync()
            samples = [[], []]
            for turn in range(rounds):
                for index in (turn % 2, 1 - turn % 2):
                    sync()
                    start = time.perf_counter()
                    for _ in range(iterations):
                        calls[index]()
                    sync()
                    samples[index].append((time.perf_counter() - start) * 1000 / iterations)
            stax, reference_ms = (statistics.median(values) for values in samples)
            item = {"name": name, "device": device, "stax_ms": stax,
                    "reference_ms": reference_ms, "speedup": reference_ms / stax}
            print(f"{device:4} {name:16} stax={stax:9.4f} ms reference={reference_ms:9.4f} ms speedup={reference_ms / stax:6.3f}x", flush=True)
        except Exception as exc:
            item = {"name": name, "device": device, "error": f"{type(exc).__name__}: {exc}"}
            print(f"{device:4} {name:16} ERROR {item['error']}", flush=True)
        results.append(item)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or min(args.iterations, args.rounds) < 1:
        parser.error("warmup must be nonnegative and iteration counts must be positive")
    results = measure(args.device, args.warmup, args.iterations, args.rounds, args.case)
    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2) + "\n")
    if any("error" in item for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
