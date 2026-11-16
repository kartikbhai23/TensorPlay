"""Measure a join followed by appending a prefix of its result."""

import argparse
import statistics
import time

import tensorplay as tp
from tensorplay.compiler.backends.stax import config


def run(device, size, iterations, rounds, compare):
    tp.set_num_threads(1)
    width = size // 2
    a = tp.ones((size, size), device=device)
    b = tp.ones((size, size), device=device)

    def region(left, right):
        joined = tp.cat([left, right], 1)
        return tp.cat([joined, joined[:, :width]], 1)

    calls = []
    for enabled in (False, True):
        graphs = []

        def record(graph):
            graphs.append([node.target for node in graph.nodes])

        with config.patch(pattern_matcher=enabled, post_grad_custom_post_pass=record):
            compiled = tp.compile(region, strict_native=True)
            output = compiled(a, b)
        rewritten = any(
            getattr(target, "_tp_lowering_function", False) for target in graphs[-1]
        )
        if rewritten != enabled or not bool((output == 1).all().item()):
            raise RuntimeError("unexpected graph or output")
        calls.append(("baseline" if not enabled else "rewritten", lambda f=compiled: f(a, b)))

    if compare:
        import torch

        torch.set_num_threads(1)
        left = torch.ones((size, size), device=device)
        right = torch.ones((size, size), device=device)

        def other_region(left, right):
            joined = torch.cat([left, right], 1)
            return torch.cat([joined, joined[:, :width]], 1)

        other = torch.compile(other_region, fullgraph=True)
        if not bool((other(left, right) == 1).all().item()):
            raise RuntimeError("incorrect comparison output")
        calls.append(("comparison", lambda: other(left, right)))

    def synchronize():
        if device == "cuda":
            tp.cuda.synchronize()

    for _ in range(20):
        for _, fn in calls:
            fn()
    synchronize()
    samples = [[] for _ in calls]
    for iteration in range(rounds):
        for offset in range(len(calls)):
            index = (iteration + offset) % len(calls)
            synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                calls[index][1]()
            synchronize()
            samples[index].append((time.perf_counter() - start) / iterations)
    for (name, _), values in zip(calls, samples):
        print(f"{device} {size}x{size} {name}: {statistics.median(values) * 1e3:.4f} ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if min(args.size, args.iterations, args.rounds) < 2:
        parser.error("size, iterations, and rounds must be at least two")
    run(args.device, args.size, args.iterations, args.rounds, args.compare)


if __name__ == "__main__":
    main()
