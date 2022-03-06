import matplotlib.pyplot as plt

def save_figure(figure: plt.Figure, filename: str, fmt: str = "eps", pad_inches=0.02):
    figure.savefig(filename, format=fmt, transparent=True, bbox_inches="tight", pad_inches=pad_inches, dpi=1200)

def save_figure_dev(figure: plt.Figure, filename: str, fmt: str = "eps", pad_inches=0.02):
    figure.savefig(filename, format=fmt, transparent=False, bbox_inches="tight", pad_inches=pad_inches)