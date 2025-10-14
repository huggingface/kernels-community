import torch
import megablocks


def randn(bs, x, y):
    out = (torch.rand(bs, x, y) - 0.5 * 2) / (y * x)
    return out.cuda().to(torch.bfloat16)


def gmm(a, b, batch_sizes, trans_b=False):
    batch_sizes = batch_sizes.cpu().numpy()

    out = []
    start = 0
    for i, size in enumerate(batch_sizes):
        rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append(a[start : start + size, :] @ rhs)
        start += size
    return torch.cat(out)


def test_gmm():
    z = 1
    m = 128
    n = 128
    k = 128
    trans_b = False
    batch_sizes_on_device = False
    # TODO: fix to enable batch_sizes_on_device
    # batch_sizes_on_device = True

    torch.manual_seed(0)
    a = randn(z, m, k).view(-1, k)
    b = randn(z, n, k) if trans_b else randn(z, k, n)
    batch_sizes = torch.tensor([m] * z)
    if batch_sizes_on_device:
        batch_sizes = batch_sizes.cuda()

    a.requires_grad_(True)
    b.requires_grad_(True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    # out = ops.gmm(a, b, batch_sizes, trans_b)
    out = megablocks.gg_ops.gmm(a, b, batch_sizes, trans_b)
    print("out", out)

    expected_out = gmm(a_ref, b_ref, batch_sizes, trans_b)

    assert torch.allclose(out, expected_out, atol=1e-3), f"Expected {expected_out}, got {out}"

    out.sum().backward()

    expected_out.sum().backward()
    assert torch.allclose(a.grad, a_ref.grad, atol=1e-3), f"Expected {a_ref.grad}, got {a.grad}"
    assert torch.allclose(b.grad, b_ref.grad, atol=1e-3), f"Expected {b_ref.grad}, got {b.grad}"
    print("Test passed successfully!")