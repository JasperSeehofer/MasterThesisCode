import logging

from parameter_estimation.schwarzschild_eccentric_flux import SchwarzschildParameterEstimation

# logging setup
logging.basicConfig(filename='logfile.log', encoding='utf-8', level=logging.DEBUG)

def main() -> None:
    """
    Run main to start the program.
    """
    parameter_estimation = SchwarzschildParameterEstimation()

    M_steps, M_differences_real, M_differences_imag = parameter_estimation.numeric_M_derivative()

    parameter_estimation._plot_M_derivative(M_steps, M_differences_real, M_differences_imag)
    

if __name__ == "__main__":
    main()