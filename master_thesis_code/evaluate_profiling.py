from master_thesis_code.cosmological_model import BayesianStatistics, Model1CrossCheck
from master_thesis_code.galaxy_catalogue.handler import GalaxyCatalogueHandler

if __name__ == "__main__":
    cosmological_model = Model1CrossCheck()
    galaxy_catalog = GalaxyCatalogueHandler(
        M_min=cosmological_model.parameter_space.M.lower_limit,
        M_max=cosmological_model.parameter_space.M.upper_limit,
        z_max=cosmological_model.max_redshift,
    )
    h_value = 0.7

    bayesian_statistics = BayesianStatistics()
    bayesian_statistics.evaluate(
        cosmological_model=cosmological_model, galaxy_catalog=galaxy_catalog, h_value=h_value
    )
