import matplotlib.pyplot as plt


def plot_round_accuracy(results: dict, title="Federated Accuracy Over Rounds"):
    """
    results = {round_num: accuracy}
    """
    rounds = sorted(results.keys())
    accs = [results[r] for r in rounds]

    plt.figure(figsize=(6, 4))
    plt.plot(rounds, accs, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
