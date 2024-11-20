import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Tuple

class ScientificPlotter:
    figure: plt.Figure
    axis: plt.Axes

    def __init__(self, figure_size: Tuple[int, int], rows: int = 1, columns: int = 1) -> None:
        self.figure, self.axis = plt.subplots(rows, columns, figsize=figure_size)

    def set_colormap_from_range(self, range: Tuple[float, float], cmap: str = 'viridis') -> None:
        norm = plt.Normalize(*range)
        self.color_map = cm.ScalarMappable(norm=norm, cmap=cmap)

    def plot(self, x: float, y: float, label: str = None) -> None:
        self.axis.plot(x, y, label=label)

    def plot_colored(self, x: float, y: float, color: float, label: str = None) -> None:
        self.axis.plot(x, y, color=self.color_map.to_rgba(color), label=label)

    def show_colorbar(self, label: str = None) -> None:
        self.color_map.set_array([])
        self.figure.colorbar(self.color_map, ax=self.axis, label=label)

    def show_and_close(self) -> None:
        self.axis.legend()
        plt.show()
        plt.close()
