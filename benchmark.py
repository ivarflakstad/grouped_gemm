import torch
import grouped_gemm as gg


if __name__ == '__main__':
    # Mixtral 8x7B sizes.
    M = 16384
    K = 4096
    N = 14336
    E = 8
    x = torch.rand(M, K, dtype=torch.bfloat16, device='cuda')
    w = torch.rand(E, K, N, dtype=torch.bfloat16, device='cuda')

    x.requires_grad_(True)
    w.requires_grad_(True)

    batch_sizes = torch.tensor([M//E]*E)

    iterations = 50
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iterations):
            out = gg.ops.gmm(x, w, batch_sizes)
            grad = out.sum().backward()

    torch.cuda.synchronize()
    device_time = prof.key_averages().total_average().device_time_total
    print(f"Total gpu time: {device_time/1000:.2f} ms")
    print(f"time per iteration: {device_time/iterations/1000:.2f} ms")
