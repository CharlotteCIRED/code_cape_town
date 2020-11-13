# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:33:37 2020

@author: Charlotte Liotta
"""

print("\n*** NEDUM-Cape-Town - Floods modelling ***\n")

import numpy as np
import scipy.io
import pandas as pd
import numpy.matlib

from data import *
from parameters_and_options import *
from compute_equilibrium import *
from export_outputs import *
from export_outputs_floods import *
from flood_outputs import *
from functions_dynamic import *
from run_simulations import *

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
param["shack_size"] = 16

#NAME OF THE SIMULATION - TO EXPORT THE RESULTS

date = "111120"
name = date + '_' + options["agri_rent"] + '_' + options["coeff_land"] + '_' + str(param["shack_size"]) + "option_transport_3"

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
#income_net_of_commuting_costs = np.load("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/precalculated_transport_v2/year_0.npy")
income_net_of_commuting_costs = scipy.io.loadmat("C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/income_net_of_commuting_cost.mat")["net_income"][:, :, 0]
param["income_year_reference"] = mean_income
data_rdp, housing_types_sp, housing_types_grid, data_sp, mitchells_plain_grid_2011, grid_formal_density_HFA, threshold_income_distribution, income_distribution, cape_town_limits = import_households_data(options)

#Macro data
interest_rate, population = import_macro_data(param)

#Land-use   
options["urban_edge"] = 1
spline_estimate_RDP, spline_land_backyard, spline_land_RDP, spline_RDP, spline_land_constraints, informal, coeff_land_backyard = import_land_use(grid, options, param, data_rdp, housing_types_grid)
number_properties_RDP = spline_estimate_RDP(0)
total_RDP = spline_RDP(0)
coeff_land = import_coeff_land(spline_land_constraints, spline_land_backyard, informal, spline_land_RDP, param, 0)
housing_limit = import_housig_limit(grid, param)
param = import_construction_parameters(param, grid, housing_types_sp, data_sp["dwelling_size"], mitchells_plain_grid_2011, grid_formal_density_HFA, coeff_land)
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
    
simul1_error, simul1_utility, simul1_households_housing_type, simul1_rent, simul1_dwelling_size, simul1_households_center, SP_code = import_basile_simulation()
   
# %% Compute initial state

print("\n*** Solver initial state ***\n")
initial_state_utility, initial_state_error, initial_state_simulated_jobs, initial_state_households_housing_types, initial_state_household_centers, initial_state_households, initial_state_dwelling_size, initial_state_housing_supply, initial_state_rent, initial_state_rent_matrix, initial_state_capital_land, initial_state_average_income, initial_state_limit_city = compute_equilibrium(fraction_capital_destroyed, amenities, param, housing_limit, population, households_per_income_class, total_RDP, coeff_land, income_net_of_commuting_costs, grid, options, agricultural_rent, interest_rate, number_properties_RDP, average_income, mean_income, income_class_by_housing_type, minimum_housing_supply, param["coeff_A"])

# %% Export outputs

os.mkdir('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name)
if options["agents_anticipate_floods"] == 0:
    fraction_capital_destroyed, depth_damage_function_structure, depth_damage_function_contents = import_floods_data()

#VALIDATION

export_housing_types(initial_state_households_housing_types, initial_state_household_centers, 1065168 * np.array([0.52, 0.08, 0.09, 0.29]), households_per_income_class, name, 'Simulation', 'Data')
export_density_rents_sizes(grid, name, data_rdp, housing_types_grid, initial_state_households_housing_types, initial_state_dwelling_size, initial_state_rent, simul1_households_housing_type, simul1_rent, simul1_dwelling_size, data_sp["dwelling_size"], SP_code)
validation_density(grid, initial_state_households_housing_types, name)
validation_density_housing_types(grid,initial_state_households_housing_types, housing_types_grid, name)
validation_housing_price(grid, initial_state_rent, interest_rate, param, center, name)

#VALIDATION - FLOODS

#Stats per housing type (flood depth and households in flood-prone areas)
floods = ['FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr', 'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr', 'FD_1000yr']
path_data = "C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/"
count_formal = housing_types_grid.formal_grid_2011 - data_rdp["count"]
count_formal[count_formal < 0] = 0
stats_per_housing_type_data = compute_stats_per_housing_type(floods, path_data, count_formal, data_rdp["count"], housing_types_grid.informal_grid_2011, housing_types_grid.backyard_grid_2011)
stats_per_housing_type_simul = compute_stats_per_housing_type(floods, path_data, initial_state_households_housing_types[0, :], initial_state_households_housing_types[3, :], initial_state_households_housing_types[2, :], initial_state_households_housing_types[1, :])
validation_flood(name, stats_per_housing_type_data, stats_per_housing_type_simul, 'Data', 'Simul')

#Damages
formal_structure_cost = compute_formal_structure_cost_method1(data_sp["price"], data_sp["dwelling_size"], SP_code, grid)
content_cost = pd.DataFrame(data = {'formal': np.empty(24014), 'informal': np.empty(24014), 'subsidized': np.empty(24014), 'backyard': np.empty(24014)})
content_cost[:] = np.nan
damages_data = compute_damages(floods, path_data, param, content_cost,
                    count_formal, data_rdp["count"], housing_types_grid.informal_grid_2011, housing_types_grid.backyard_grid_2011,
                    formal_structure_cost, depth_damage_function_structure, depth_damage_function_contents)
    
formal_structure_cost = compute_formal_structure_cost_method2(initial_state_rent, param, interest_rate, coeff_land, initial_state_households_housing_types)    
content_cost = compute_content_cost(initial_state_household_centers, income_net_of_commuting_costs, param, fraction_capital_destroyed, initial_state_rent, initial_state_dwelling_size, interest_rate)
damages_simul = compute_damages(floods, path_data, param, content_cost,
                    initial_state_households_housing_types[0, :], initial_state_households_housing_types[3, :], initial_state_households_housing_types[2, :], initial_state_households_housing_types[1, :],
                    formal_structure_cost, depth_damage_function_structure, depth_damage_function_contents)
    
plot_damages(damages_simul, name)
compare_damages(damages_simul, damages_data, "Simulation", "Data", name)

# %% Scenarios

#Compute scenarios
t = np.arange(0, 30)
simulation_households_center, simulation_households_housing_type, simulation_dwelling_size, simulation_rent, simulation_households, simulation_error, simulation_housing_supply, simulation_utility, simulation_deriv_housing, simulation_T = run_simulation(t, options, income_2011, param, grid, initial_state_utility, initial_state_error, initial_state_households, initial_state_households_housing_types, initial_state_housing_supply, initial_state_household_centers, initial_state_average_income, initial_state_rent, initial_state_dwelling_size, fraction_capital_destroyed, amenities, housing_limit, spline_estimate_RDP, spline_land_constraints, spline_land_backyard, spline_land_RDP, informal, income_class_by_housing_type)

#Export outputs
name = name + "_scenarios"
export_housing_types(simulation_households_housing_type[0, :, :], simulation_households_center[0, :, :], simulation_households_housing_type[29, :, :], simulation_households_center[29, :, :], name, '2011', '2040')
simul_2011 = simulation_households_housing_type[0, :, :]
simul_2040 = simulation_households_housing_type[29, :, :]
stats_per_housing_type_2011 = compute_stats_per_housing_type(floods, path_data, simul_2011[0, :], simul_2011[3, :], simul_2011[2, :], simul_2011[1, :])
stats_per_housing_type_2040 = compute_stats_per_housing_type(floods, path_data, simul_2040[0, :], simul_2040[3, :], simul_2040[2, :], simul_2040[1, :])
validation_flood(name, stats_per_housing_type_2011, stats_per_housing_type_2040, '2011', '2040')
#compare_damages(damages_simul, damages_data, "Simulation", "Data", name)












