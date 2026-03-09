from typing import Any

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class ScientificPlotter:
    figure: plt.Figure
    axis: plt.Axes

    def __init__(self, figure_size: tuple[int, int], rows: int = 1, columns: int = 1) -> None:
        self.figure, self.axis = plt.subplots(rows, columns, figsize=figure_size)
        plt.rcParams["text.usetex"] = True

    def set_colormap_from_range(self, range: tuple[float, float], cmap: str = "viridis") -> None:
        norm = plt.Normalize(*range)
        self.color_map = cm.ScalarMappable(norm=norm, cmap=cmap)

    def scatter(
        self,
        x: float | npt.NDArray[np.floating[Any]],
        y: float | npt.NDArray[np.floating[Any]],
        label: str | None = None,
        kwargs: dict[str, Any] = {},
    ) -> None:
        self.axis.scatter(x, y, label=label, **kwargs)

    def plot(
        self,
        x: float | npt.NDArray[np.floating[Any]],
        y: float | npt.NDArray[np.floating[Any]],
        label: str | None = None,
        kwargs: dict[str, Any] = {},
    ) -> None:
        self.axis.plot(x, y, label=label, **kwargs)

    def plot_colored(
        self,
        x: float | npt.NDArray[np.floating[Any]],
        y: float | npt.NDArray[np.floating[Any]],
        color: float | npt.NDArray[np.floating[Any]],
        label: str | None = None,
        line_style: str | None = None,
        kwargs: dict[str, Any] = {},
    ) -> None:
        self.axis.plot(
            x,
            y,
            color=self.color_map.to_rgba(color),  # type: ignore[arg-type]
            label=label,
            linestyle=line_style,
            **kwargs,
        )

    def show_colorbar(self, label: str | None = None) -> None:
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
