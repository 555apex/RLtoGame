"""可独立运行的梯度与 TD 手算示例，不执行游戏训练。"""

import torch

from snake.agents.dqn import dqn_targets


def main():
    weight = torch.tensor(1.0, requires_grad=True)
    bias = torch.tensor(0.0, requires_grad=True)
    prediction = weight * 2 + bias
    loss = (prediction - 3).square() / 2
    loss.backward()
    print(f"Before: prediction={prediction.item():.2f}, grad_w={weight.grad.item():.2f}, grad_b={bias.grad.item():.2f}")
    with torch.no_grad():
        weight -= 0.1 * weight.grad
        bias -= 0.1 * bias.grad
    print(f"After: w={weight.item():.2f}, b={bias.item():.2f}, prediction={(weight * 2 + bias).item():.2f}")
    values = dqn_targets(torch.tensor([1., 1., 1.]), torch.tensor([False, True, False]),
                         torch.tensor([[1., 3., 2.]] * 3), gamma=0.9)
    print("TD targets (ordinary, terminal, time-truncated):", [round(x, 3) for x in values.tolist()])


if __name__ == "__main__":
    main()
