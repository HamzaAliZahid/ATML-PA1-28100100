import matplotlib.pyplot as plt


lambda_values = [0.1, 1.0, 10.0]

target_accuracy = [
    0.7427,
    0.7878,
    0.1300
]

plt.figure(figsize=(7, 5))

plt.plot(
    lambda_values,
    target_accuracy,
    marker="o"
)

plt.xscale("log")

plt.xlabel("MMD λ")
plt.ylabel("Sketch Target Accuracy")

plt.title("DAN Controlled Study: Effect of MMD Alignment Strength")

plt.xticks(
    lambda_values,
    ["0.1", "1", "10"]
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "task2/dan_controlled_study.png",
    dpi=300
)

plt.show()