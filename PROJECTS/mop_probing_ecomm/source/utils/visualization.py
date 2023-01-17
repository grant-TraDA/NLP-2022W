import matplotlib.pyplot as plt


def plot_histogram(values, bins):
    _ = plt.hist(values, bins=bins)
    plt.title(f"Histogram with {bins} bins")
    plt.show()