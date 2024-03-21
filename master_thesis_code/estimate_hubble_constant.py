import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

from master_thesis_code.constants import C
from master_thesis_code.galaxy_catalogue.handler import GalaxyCatalogueHandler, InternalCatalogColumns

_LOGGER = logging.getLogger()

class HubbleConstantEstimation:

    def __init__(self):
        self.cramer_rao_bounds = pd.read_csv("./simulations/cramer_rao_bounds_unbiased.csv")

    
    def bayesian_evaluation(self, galaxy_catalog: GalaxyCatalogueHandler) -> None:
        hubble_estimation = []
        hubble_estimation_with_bh_mass = []

        print(max(galaxy_catalog.reduced_galaxy_catalog[InternalCatalogColumns.LUMINOSITY_DISTANCE]))

        for index, cramer_rao_bound in self.cramer_rao_bounds.iterrows():
            parameters = {
                "dist": cramer_rao_bound["dist"],
                "dist_error": cramer_rao_bound["delta_dist_delta_dist"],
                "phi": cramer_rao_bound["phiS"],
                "phi_error": cramer_rao_bound["delta_phiS_delta_phiS"],
                "theta": cramer_rao_bound["qS"],
                "theta_error": cramer_rao_bound["delta_qS_delta_qS"],
                "M_z": cramer_rao_bound["M"],
                "M_z_error": cramer_rao_bound["delta_M_delta_M"],
            }

            possible_hosts, possible_hosts_with_BH_mass = galaxy_catalog.get_possible_hosts(**parameters)

            if len(possible_hosts) == 0:
                _LOGGER.info("no possible hosts found, skip event.")
                continue

            H, H_error = self.get_hubble_constants(possible_hosts, parameters["dist"], parameters["dist_error"])
            hubble_estimation.append([H, H_error])

            if len(possible_hosts_with_BH_mass) == 0:
                _LOGGER.info("no possible hosts for bh evaluation found, skip event.")
                continue

            HBH, HBH_error = self.get_hubble_constants(possible_hosts_with_BH_mass, parameters["dist"], parameters["dist_error"])
            hubble_estimation_with_bh_mass.append([HBH, HBH_error])

        
        # plot the estimations
            
        hubble_estimation = np.array(hubble_estimation)
        hubble_estimation_with_bh_mass = np.array(hubble_estimation_with_bh_mass)
            
        print(hubble_estimation, hubble_estimation_with_bh_mass)

        mean_hubble = np.mean(hubble_estimation[:, 0])
        mean_error = np.sqrt(np.sum(hubble_estimation[:][1]))/len(hubble_estimation[:, 1])
        std = np.std(hubble_estimation[:, 0])

        _LOGGER.info(f"Hubble constant estimation without BH mass: H = {mean_hubble} +/- {mean_error}, std = {std}")

        mean_hubble_with_BH = np.mean(hubble_estimation_with_bh_mass[:, 0])
        mean_error_with_BH = np.sqrt(np.sum(hubble_estimation_with_bh_mass[:, 1]))/len(hubble_estimation_with_bh_mass[:, 1])
        std_with_BH = np.std(hubble_estimation_with_bh_mass[:, 0])

        _LOGGER.info(f"Hubble constant estimation without BH mass: H = {mean_hubble_with_BH} +/- {mean_error_with_BH}, std = {std_with_BH}")

        # plot histrogramm
        plt.figure(figsize=(16, 9))
        plt.hist(hubble_estimation[:, 0], histtype="step")
        plt.xlabel("hubble estimation")
        plt.ylabel("#")
        plt.savefig(f"./evaluation/plots/hubble_estimations.png")
        plt.close()

        plt.figure(figsize=(16, 9))
        plt.hist(hubble_estimation_with_bh_mass[:, 0], histtype="step")
        plt.xlabel("hubble estimation")
        plt.ylabel("#")
        plt.savefig(f"./evaluation/plots/hubble_estimations_with_bh.png")
        plt.close()
        


    def get_hubble_constants(self, possible_hosts: pd.DataFrame, dist: float, dist_error: float) -> tuple:
        hubble_estimations = []
        hubble_errors = []

        for index, host in possible_hosts.iterrows():
            z = host[InternalCatalogColumns.REDSHIFT]
            z_error = host[InternalCatalogColumns.REDSHIFT_ERROR]

            if (not isinstance(z, float)) or (not isinstance(z_error, float)) :
                _LOGGER.info("No redshift and error given for host, so no hubble estimation is made.")
                continue

            H = z/dist*C*1e-3
            H_error = C/dist*np.sqrt(z_error**2 + (z*dist_error/dist)**2)*1e-3
            hubble_estimations.append(H)
            hubble_errors.append(H_error)

        print(hubble_estimations)
        print(hubble_errors)

        mean_hubble = np.mean(np.array(hubble_estimations))
        mean_error = np.sqrt(np.sum(np.array(hubble_errors)**2))/float(len(hubble_errors))
        return (mean_hubble, mean_error)




