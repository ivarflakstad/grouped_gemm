import torch
from typing import Tuple
import grouped_gemm_backend


def _allocate_output(a, b, batch_sizes, trans_a, trans_b):
    assert not (trans_a and trans_b)
    assert batch_sizes.ndim == 1, "Expected 1d tensor for batch_sizes"
    assert a.ndim == 2, "Expected 2d tensor for 'a'"
    assert b.ndim == (2 if trans_a else 3)

    assert a.is_contiguous()
    assert b.is_contiguous()
    assert batch_sizes.is_contiguous()

    #return torch.empty((M, N), device=a.device, dtype=a.dtype)

    shape = (
        (batch_sizes.shape[0], a.shape[1], b.shape[1])
        if trans_a else
        (a.shape[0], (b.shape[1] if trans_b else b.shape[2]))
    )
    print(shape)
    return torch.empty(*shape, device=a.device, dtype=a.dtype)

def gmm(a, b, batch_sizes, trans_a=False, trans_b=False, c=None):
    if c is None:
        c = _allocate_output(a, b, batch_sizes, trans_a, trans_b)
    grouped_gemm_backend.ck_gmm(a, b, c, batch_sizes, trans_a, trans_b)
    return c

def test_grouped_gemm_bf16(
    shape: Tuple[int, int, int, int],
    device: torch.device,
) -> None:
    G, M, N, K = shape
    print(f"G: {G}, M: {M}, N: {N}, K: {K}")
    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(G, N, K, dtype=torch.bfloat16, device=device)
    m_ends, _ = torch.sort(
        torch.randint(
            low=0, high=M, size=[G - 1], device=device, dtype=torch.int32
        )
    )
    m_ends = m_ends.tolist()
    m_starts = [0] + m_ends
    m_ends = m_ends + [M]
    m_sizes = torch.tensor(
        [m_ends[i] - m_starts[i] for i in range(G)], device=device
    ).to(torch.int64)

    result = gmm(
        a,
        b,
        m_sizes,
    )
    assert result.shape == (M, N), f"Expected shape {(M, N)}, got {result.shape}"

    expected_result = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
    for g in range(G):
        m_start = m_starts[g]
        m_end = m_ends[g]
        expected_result[m_start:m_end, :] = (
            a[m_start:m_end, :] @ b[g * N : (g + 1) * N, :].mT
        )

    torch.testing.assert_close(result, expected_result, atol=1e-5, rtol=1.6e-2)


if __name__ == "__main__":
    # Test basic connection
    result = grouped_gemm_backend.test_connection()
    print(result)  # Should print a 2x2 tensor filled with 123.0

    for G in (1, 4, 16):
        for M in (64, 512):
            test_grouped_gemm_bf16((G, M, 256, 256), torch.device("cuda"))
