import matplotlib.pyplot as plt

xs = ["A - Inside", "B - Contact", ]
ys = [.14, .42]

plt.bar(xs, ys)
plt.ylim(0, 1)
plt.title("M - SI")
plt.savefig("Inside_ablate")