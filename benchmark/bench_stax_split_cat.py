"""Compare execution with and without the split-and-join graph rewrite."""

import argparse
import statistics
import time

import tensorplay as tp
from tensorplay.compiler.backends.stax import config


def benchmark(device, size, iterations, rounds):
    x = tp.ones((size, size), device=device)
    left_width = size // 2

    def region(value):
        left, right = value.split([left_width, size - left_width], dim=1)
        return tp.cat([left, right], dim=1) + 1

    compiled = []
    for enabled in (False, True):
        graphs = []

        def record(graph):
            graphs.append([node.target for node in graph.nodes])

        with config.patch(pattern_matcher=enabled, post_grad_custom_post_pass=record):
            fn = tp.compile(region, strict_native=True)
            output = fn(x)
        if not bool((output == 2).all().item()):
            raise RuntimeError("incorrect output")
        if not graphs or (tp.ops.tp.cat.default in graphs[-1]) == enabled:
            raise RuntimeError("the expected graph rewrite did not occur")
        compiled.append(fn)

    def synchronize():
        if device == "cuda":
            tp.cuda.synchronize()

    for _ in range(20):
        for fn in compiled:
            fn(x)
    synchronize()
    samples = [[], []]
    for iteration in range(rounds):
        for index in (iteration % 2, 1 - iteration % 2):
            synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                compiled[index](x)
            synchronize()
            samples[index].append((time.perf_counter() - start) / iterations)
    baseline, rewritten = (statistics.median(sample) for sample in samples)
    print(
        f"{device} {size}x{size}: baseline={baseline * 1e3:.4f} ms, "
        f"rewritten={rewritten * 1e3:.4f} ms, speedup={baseline / rewritten:.3f}x"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()
    if min(args.size, args.iterations, args.rounds) < 1:
        parser.error("size, iterations, and rounds must be positive")
    tp.set_num_threads(1)
    benchmark(args.device, args.size, args.iterations, args.rounds)


if __name__ == "__main__":
    main()
