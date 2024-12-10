import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import Tuple


class ScientificPlotter:
    figure: plt.Figure
    axis: plt.Axes

    def __init__(
        self, figure_size: Tuple[int, int], rows: int = 1, columns: int = 1
    ) -> None:
        self.figure, self.axis = plt.subplots(rows, columns, figsize=figure_size)
        plt.rcParams["text.usetex"] = True

    def set_colormap_from_range(
        self, range: Tuple[float, float], cmap: str = "viridis"
    ) -> None:
        norm = plt.Normalize(*range)
        self.color_map = cm.ScalarMappable(norm=norm, cmap=cmap)

    def plot(self, x: float, y: float, label: str = None, kwargs: dict = {}) -> None:
        self.axis.plot(x, y, label=label, **kwargs)

    def plot_colored(
        self,
        x: float,
        y: float,
        color: float,
        label: str = None,
        line_style: str = None,
        kwargs: dict = {},
    ) -> None:
        self.axis.plot(
            x,
            y,
            color=self.color_map.to_rgba(color),
            label=label,
            linestyle=line_style,
            **kwargs,
        )

    def show_colorbar(self, label: str = None) -> None:
        self.color_map.set_array([])
        self.figure.colorbar(self.color_map, ax=self.axis, label=label)

    def show_and_close(self) -> None:
        self.axis.legend()
        plt.show()
        plt.close(self.figure)

    def save_as_svg(self, file_path: str, dpi: int) -> None:
        self.axis.legend()
        self.figure.savefig(file_path, format="svg", dpi=dpi)
        plt.close(self.figure)

    def plot_example(self) -> None:
        x = np.linspace(0, 5, 1000)
        number_of_plots = 10
        self.set_colormap_from_range((0, number_of_plots - 1))
        for i in range(number_of_plots):
            y = x * i
            self.plot_colored(x, y, i, label=rf"${i} \cdot x$")
        self.show_colorbar("Colorbar label")
        self.save_as_svg(file_path="example.svg", dpi=300)


if __name__ == "__main__":
    plotter = ScientificPlotter((16, 9))
    plotter.plot_example()
