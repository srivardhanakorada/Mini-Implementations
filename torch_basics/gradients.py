import torch

torch.manual_seed(0)
x = torch.rand(2,3,2,requires_grad=True)
y = (x**2).sum()

# via auto_grad
y.backward()
grad_auto = x.grad.detach().clone()

# via manual_grad
grad_manual = 2 * x.detach()

# Compare
max_abs_diff = (grad_auto - grad_manual).abs().max().item()
print("Max abs diff:", max_abs_diff)

assert max_abs_diff < 1e-6, "Autograd sanity check failed!"