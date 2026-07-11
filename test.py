from value import value
from neural_net import neuron , layer, mlp
def main():
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]  # desired targets

    n = mlp(3, [4, 4, 1])

    for k in range(20):
        # forward pass
        y_pred = [n(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, y_pred))

        # zero grad
        for p in n.parameters():
            p.grad = 0.0

        # backward pass
        loss.backward()

        # update
        for p in n.parameters():
            p.data -= 0.05 * p.grad

        print(f"step {k}: loss = {loss.data:.4f}")

    print("\nFinal predictions:")
    y_pred = [n(x) for x in xs]
    for ygt, yout in zip(ys, y_pred):
        print(f"target: {ygt:+.1f}  predicted: {yout.data:+.4f}")


if __name__ == "__main__":
    main()