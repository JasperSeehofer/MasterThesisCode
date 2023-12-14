import pandas as pd
import os
from itertools import combinations_with_replacement
from master_thesis_code.datamodels.parameter_space import parameters_configuration, get_parameter_configuration
import matplotlib.pyplot as plt

class DataEvaluation:

    def __init__(self, path_to_cramer_rao_bounds_file: str = "./simulations/cramer_rao_bounds.csv"):
        self._cramer_rao_bounds = pd.read_csv(path_to_cramer_rao_bounds_file)

    def visualize(self) -> None:
        # ensure directory is given
        figures_directory = f"evaluation/"
        if not os.path.isdir(figures_directory):
            os.makedirs(figures_directory)

        parameter_symbol_list = [parameter.symbol for parameter in parameters_configuration]
        parameter_combinations = combinations_with_replacement(parameter_symbol_list, 2)
        
        mean_cramer_rao_bounds = pd.DataFrame(index=parameter_symbol_list, columns=parameter_symbol_list)
        
        for a,b in parameter_combinations:
            column_name = f"delta_{b}_delta_{a}"
            mean_cramer_rao_bounds.at[a, b] = self._cramer_rao_bounds[column_name].mean()
        
        mean_cramer_rao_bounds.to_excel(f"{figures_directory}mean_bounds.xlsx")

        # interesting plots
        bounds_parameters = ["M", "qS", "phiS", "qK", "phiK"]
        dependency_parameters = ["qS", "phiS", "qK", "phiK", "Phi_phi0"]

        weird_ones = self._cramer_rao_bounds[self._cramer_rao_bounds["delta_phiS_delta_phiS"] > 10**16]
        weird_ones.to_excel(f"{figures_directory}weird_ones.xlsx")


        for bounds_parameter in bounds_parameters:
            bounds_column_name = f"delta_{bounds_parameter}_delta_{bounds_parameter}"
            for dependency_parameter in parameter_symbol_list:
                bound_parameter_config = get_parameter_configuration(bounds_parameter)
                dependency_parameter_config = get_parameter_configuration(dependency_parameter)
                plt.figure(figsize=(16,9))
                plt.scatter(
                    self._cramer_rao_bounds[dependency_parameter], 
                    self._cramer_rao_bounds[bounds_column_name], 
                    label=f"error of {bounds_parameter}"
                )
                plt.scatter(
                    weird_ones[dependency_parameter], 
                    weird_ones[bounds_column_name],
                    c="red",
                    marker="x", 
                    label=f"weird ones"
                )
                plt.xlabel(f"{dependency_parameter} [{dependency_parameter_config.unit}]")
                plt.ylabel(f"error bounds {bounds_parameter} [{bound_parameter_config.unit}]")
                plt.legend()
                plt.yscale("log")
                plt.savefig(f"{figures_directory}plots/error_{bounds_parameter}_{dependency_parameter}.png")
                plt.close()
