# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:29:30 2020

@author: Charlotte Liotta
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:40:44 2020

@author: Charlotte Liotta
"""

import numpy as np
import scipy.io
import pandas as pd
import numpy.matlib

from data import *
from parameters_and_options import *
from compute_equilibrium import *
from export_outputs import *
from export_outputs_floods import *
from functions_dynamic import *
from run_simulations import *

print('**************** NEDUM-Cape-Town - Calibration of the informal housing amenity parameters ****************')

# %% Choose parameters and options

print("\n*** Load parameters and options ***\n")

# %% Import parameters and options

print("\n*** Load parameters and options ***\n")

#IMPORT PARAMETERS AND OPTIONS

options = import_options()
param = import_param(options)
t = np.arange(0, 1)

#OPTIONS FOR THIS SIMULATION

options["agents_anticipate_floods"] = 1
options["agri_rent"] = "low"
options["coeff_land"] = 'old'
param["subsidized_structure_value"] = 150000
param["fraction_z_dwellings"] = 0.49

# %% Load data

print("\n*** Load data ***\n")

#DATA

#Grid
grid, center = import_grid()
amenities = import_amenities()

#Households data
income_class_by_housing_type = import_hypothesis_housing_type()
income_2011 = pd.read_csv('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Basile data/Income_distribution_2011.csv')
mean_income = np.sum(income_2011.Households_nb * income_2011.INC_med) / sum(income_2011.Households_nb)
households_per_income_class, average_income = import_income_classes_data(param, income_2011)
income_mult = average_income / mean_income
income_net_of_commuting_costs = np.load("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/precalculated_transport/year_0.npy")
param["income_year_reference"] = mean_income
data_rdp, housing_types_sp, housing_types_grid, dwelling_size_sp, mitchells_plain_grid_2011, grid_formal_density_HFA, sp_price = import_households_data(options)

#Macro data
interest_rate, population = import_macro_data(param)

#Land-use   
options["urban_edge"] = 1
spline_estimate_RDP, spline_land_backyard, spline_land_RDP, spline_RDP, spline_land_constraints, informal, coeff_land_backyard = import_land_use(grid, options, param, data_rdp, housing_types_grid)
number_properties_RDP = spline_estimate_RDP(0)
total_RDP = spline_RDP(0)
coeff_land = import_coeff_land(spline_land_constraints, spline_land_backyard, informal, spline_land_RDP, param, 0)
housing_limit = import_housig_limit(grid, param)
param = import_construction_parameters(param, grid, housing_types_sp, dwelling_size_sp, mitchells_plain_grid_2011, grid_formal_density_HFA, coeff_land)
minimum_housing_supply = param["minimum_housing_supply"]
if options["agri_rent"] == "low":
    agricultural_rent = param["agricultural_rent_2011"] ** (param["coeff_a"]) * (param["depreciation_rate"] + interest_rate) / (param["coeff_A"] * param["coeff_b"] ** param["coeff_b"])
elif options["agri_rent"] == "high":
    agricultural_rent = param["agricultural_rent_2011"]

#FLOOD DATA
if options["agents_anticipate_floods"] == 1:
    fraction_capital_destroyed, depth_damage_function_structure, depth_damage_function_contents = import_floods_data()
elif options["agents_anticipate_floods"] == 0:
    fraction_capital_destroyed = pd.DataFrame()
    fraction_capital_destroyed["structure"] = np.zeros(24014)
    fraction_capital_destroyed["contents"] = np.zeros(24014)
    
#simul1_error, simul1_utility, simul1_households_housing_type, simul1_rent, simul1_dwelling_size, simul1_households_center, SP_code = import_basile_simulation()

# %% Run initial state for several values of amenities

#List of parameters
list_amenity_backyard = np.arange(0.65, 0.81, 0.01)
list_amenity_settlement = np.arange(0.65, 0.81, 0.01)
housing_type_total = np.zeros((3, 4, len(list_amenity_backyard) * len(list_amenity_settlement)))

sum_housing_types = lambda initial_state_households_housing_types : np.nansum(initial_state_households_housing_types, 1)
index = 0
for i in range(0, len(list_amenity_backyard)):
    for j in range(0, len(list_amenity_settlement)):
        param["amenity_backyard"] = list_amenity_backyard[i]
        param["amenity_settlement"] = list_amenity_settlement[j]
        initial_state_utility, initial_state_error, initial_state_simulated_jobs, initial_state_households_housing_types, initial_state_household_centers, initial_state_households, initial_state_dwelling_size, initial_state_housing_supply, initial_state_rent, initial_state_rent_matrix, initial_state_capital_land, initial_state_average_income, initial_state_limit_city = compute_equilibrium(fraction_capital_destroyed, amenities, param, housing_limit, population, households_per_income_class, total_RDP, coeff_land, income_net_of_commuting_costs, grid, options, agricultural_rent, interest_rate, number_properties_RDP, average_income, mean_income, income_class_by_housing_type, minimum_housing_supply, param["coeff_A"])
        housing_type_total[0, : , index] = param["amenity_backyard"]
        housing_type_total[1, : , index] = param["amenity_settlement"]
        housing_type_total[2, : , index] = sum_housing_types(initial_state_households_housing_types)
        index = index + 1

print('*** End of simulations for chosen parameters ***')

# %% Pick best solution

housing_type_data = np.array([np.nansum(housing_types_grid.formal_grid_2011) - np.nansum(initial_state_households_housing_types[3,:]), np.nansum(housing_types_grid.backyard_grid_2011), np.nansum(housing_types_grid.informal_grid_2011), np.nansum(housing_types_grid.formal_grid_2011 + housing_types_grid.backyard_grid_2011 + housing_types_grid.informal_grid_2011)])
housing_type_data = np.array([1070000 * 0.52, 1070000 * 0.08, 1070000 * 0.09, 1070000])

distance_share = np.abs(housing_type_total[2, 0:3, :] - housing_type_data[0:3, None])
distance_share_score = distance_share[1,:] #+ 0.8 * distance_share[0,:] + 0.8 * distance_share[2,:]
which = np.argmin(distance_share_score)
min_score = np.nanmin(distance_share_score)
calibrated_amenities = housing_type_total[0:2, 0, which]
housing_type_total[2, :, which]

param["amenity_backyard"] = calibrated_amenities[0]
param["amenity_settlement"] = calibrated_amenities[1]