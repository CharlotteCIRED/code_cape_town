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

from inputs.data import *
from inputs.parameters_and_options import *
from equilibrium.compute_equilibrium import *
from outputs.export_outputs import *
from outputs.export_outputs_floods import *
from outputs.flood_outputs import *
from equilibrium.functions_dynamic import *
from equilibrium.run_simulations import *
from inputs.WBUS2_depth import *

# %% Import parameters and options

print("\n*** Load parameters and options ***\n")

#IMPORT PARAMETERS AND OPTIONS
options = import_options()
param = import_param(options)
t = np.arange(0, 1)

#OPTIONS FOR THIS SIMULATION
options["pluvial"] = 0
options["informal_land_constrained"] = 0
param["threshold"] = 130
 
#NAME OF THE SIMULATION - TO EXPORT THE RESULTS
date = '20201216'
name = date + '_' + str(options["pluvial"]) + '_' + str(options["informal_land_constrained"])

# %% Load data

print("\n*** Load data ***\n")

#DATA

#Grid
grid, center = import_grid()
amenities = import_amenities()

#Households and income data
income_class_by_housing_type = import_hypothesis_housing_type()
income_2011 = pd.read_csv('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Basile data/Income_distribution_2011.csv')
mean_income = np.sum(income_2011.Households_nb * income_2011.INC_med) / sum(income_2011.Households_nb)
households_per_income_class, average_income = import_income_classes_data(param, income_2011)
income_mult = average_income / mean_income
income_net_of_commuting_costs = np.load("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/precalculated_transport/year_0.npy")
param["income_year_reference"] = mean_income
data_rdp, housing_types_sp, data_sp, mitchells_plain_grid_2011, grid_formal_density_HFA, threshold_income_distribution, income_distribution, cape_town_limits = import_households_data(options)
housing_types = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/housing_types_grid_sal.xlsx')
housing_types[np.isnan(housing_types)] = 0

#Macro data
interest_rate, population = import_macro_data(param)
total_RDP = 194258
total_formal = 626770
total_informal = 143765
total_backyard = 91132

#Land-use   
options["urban_edge"] = 1
spline_estimate_RDP, spline_land_backyard, spline_land_RDP, spline_RDP, spline_land_constraints, spline_land_informal, coeff_land_backyard = import_land_use(grid, options, param, data_rdp, housing_types, total_RDP)
number_properties_RDP = spline_estimate_RDP(0)
coeff_land = import_coeff_land(spline_land_constraints, spline_land_backyard, spline_land_informal, spline_land_RDP, param, 0)
housing_limit = import_housig_limit(grid, param)
param = import_construction_parameters(param, grid, housing_types_sp, data_sp["dwelling_size"], mitchells_plain_grid_2011, grid_formal_density_HFA, coeff_land)
minimum_housing_supply = param["minimum_housing_supply"]
agricultural_rent = param["agricultural_rent_2011"] ** (param["coeff_a"]) * (param["depreciation_rate"] + interest_rate) / (param["coeff_A"] * param["coeff_b"] ** param["coeff_b"])
print(sum(spline_land_informal(0)))

#Floods
param = infer_WBUS2_depth(housing_types, param)
if options["agents_anticipate_floods"] == 1:
    fraction_capital_destroyed, content_damages, structural_damages_type4b, structural_damages_type2, structural_damages_type3a = import_floods_data(options, param)
elif options["agents_anticipate_floods"] == 0:
    fraction_capital_destroyed = pd.DataFrame()
    fraction_capital_destroyed["structure_formal_2"] = np.zeros(24014)
    fraction_capital_destroyed["structure_formal_1"] = np.zeros(24014)
    fraction_capital_destroyed["structure_subsidized_2"] = np.zeros(24014)
    fraction_capital_destroyed["structure_subsidized_1"] = np.zeros(24014)
    fraction_capital_destroyed["contents_formal"] = np.zeros(24014)
    fraction_capital_destroyed["contents_informal"] = np.zeros(24014)
    fraction_capital_destroyed["contents_subsidized"] = np.zeros(24014)
    fraction_capital_destroyed["contents_backyard"] = np.zeros(24014)
    fraction_capital_destroyed["structure_backyards"] = np.zeros(24014)
    fraction_capital_destroyed["structure_informal_settlements"] = np.zeros(24014)

# %% Calibration

#General calibration
list_amenity_backyard = np.arange(0.70, 0.90, 0.01)
list_amenity_settlement = np.arange(0.67, 0.87, 0.01)
housing_type_total = pd.DataFrame(np.array(np.meshgrid(list_amenity_backyard, list_amenity_settlement)).T.reshape(-1,2))
housing_type_total.columns = ["param_backyard", "param_settlement"]
housing_type_total["formal"] = np.zeros(len(housing_type_total.param_backyard))
housing_type_total["backyard"] = np.zeros(len(housing_type_total.param_backyard))
housing_type_total["informal"] = np.zeros(len(housing_type_total.param_backyard))
housing_type_total["subsidized"] = np.zeros(len(housing_type_total.param_backyard))

sum_housing_types = lambda initial_state_households_housing_types : np.nansum(initial_state_households_housing_types, 1)

for i in range(0, len(list_amenity_backyard)):
    for j in range(0, len(list_amenity_settlement)):
        param["amenity_backyard"] = list_amenity_backyard[i]
        param["amenity_settlement"] = list_amenity_settlement[j]
        param["pockets"] = np.ones(24014) * param["amenity_settlement"]
        param["backyard_pockets"] = np.ones(24014) * param["amenity_backyard"]
        initial_state_utility, initial_state_error, initial_state_simulated_jobs, initial_state_households_housing_types, initial_state_household_centers, initial_state_households, initial_state_dwelling_size, initial_state_housing_supply, initial_state_rent, initial_state_rent_matrix, initial_state_capital_land, initial_state_average_income, initial_state_limit_city = compute_equilibrium(fraction_capital_destroyed, amenities, param, housing_limit, population, households_per_income_class, total_RDP, coeff_land, income_net_of_commuting_costs, grid, options, agricultural_rent, interest_rate, number_properties_RDP, average_income, mean_income, income_class_by_housing_type, minimum_housing_supply, param["coeff_A"])
        housing_type_total.loc[(housing_type_total.param_backyard == param["amenity_backyard"]) & (housing_type_total.param_settlement == param["amenity_settlement"]), 2:6] = sum_housing_types(initial_state_households_housing_types)
    
housing_type_data = np.array([total_formal, total_backyard, total_informal, total_RDP])

distance_share = np.abs(housing_type_total.iloc[:, 2:5] - housing_type_data[None, 0:3])
distance_share_score = distance_share.iloc[:,1] + distance_share.iloc[:,2] +  distance_share.iloc[:,0]
which = np.argmin(distance_share_score)
min_score = np.nanmin(distance_share_score)
calibrated_amenities = housing_type_total.iloc[which, 0:2]

#0.88 et 0.85

param["amenity_backyard"] = calibrated_amenities[0]
param["amenity_settlement"] = calibrated_amenities[1]

param["amenity_backyard"] = 0.89
param["amenity_settlement"] = 0.86

#Location-based calibration
index = 0
index_max = 400
metrics = np.zeros(index_max)
param["pockets"] = np.zeros(24014) + param["amenity_settlement"]
save_param_informal_settlements = np.zeros((index_max, 24014))
metrics_is = np.zeros(index_max)
param["backyard_pockets"] = np.zeros(24014) + param["amenity_backyard"]
save_param_backyards = np.zeros((index_max, 24014))
metrics_ib = np.zeros(index_max)
initial_state_utility, initial_state_error, initial_state_simulated_jobs, initial_state_households_housing_types, initial_state_household_centers, initial_state_households, initial_state_dwelling_size, initial_state_housing_supply, initial_state_rent, initial_state_rent_matrix, initial_state_capital_land, initial_state_average_income, initial_state_limit_city = compute_equilibrium(fraction_capital_destroyed, amenities, param, housing_limit, population, households_per_income_class, total_RDP, coeff_land, income_net_of_commuting_costs, grid, options, agricultural_rent, interest_rate, number_properties_RDP, average_income, mean_income, income_class_by_housing_type, minimum_housing_supply, param["coeff_A"])


for index in range(0, index_max):
    
    print("******************************* NEW ITERATION **************************************")
    print("INDEX  = " + str(index))
    
    #IS
    diff_is = np.zeros(24014)
    for i in range(0, 24014):
        diff_is[i] = (housing_types.informal_grid[i]) - (initial_state_households_housing_types[2,:][i])
        adj = (diff_is[i] / (50000))
        save_param_informal_settlements[index, :] = param["pockets"]
        param["pockets"][i] = param["pockets"][i] + adj
    metrics_is[index] = sum(np.abs(diff_is))
    param["pockets"][param["pockets"] < 0.05] = 0.05
    param["pockets"][param["pockets"] > 0.99] = 0.99
    
    #IB
    diff_ib = np.zeros(24014)
    for i in range(0, 24014):
        diff_ib[i] = ((housing_types.backyard_informal_grid[i] + housing_types.backyard_formal_grid[i]) - (initial_state_households_housing_types[1,:][i]))
        adj = (diff_ib[i] / 50000)
        save_param_backyards[index, :] = param["backyard_pockets"]
        param["backyard_pockets"][i] = param["backyard_pockets"][i] + adj
    metrics_ib[index] = sum(np.abs(diff_ib))
    param["backyard_pockets"][param["backyard_pockets"] < 0.05] = 0.05
    param["backyard_pockets"][param["backyard_pockets"] > 0.99] = 0.99
    
    metrics[index] = metrics_is[index] + metrics_ib[index]
    
    initial_state_utility, initial_state_error, initial_state_simulated_jobs, initial_state_households_housing_types, initial_state_household_centers, initial_state_households, initial_state_dwelling_size, initial_state_housing_supply, initial_state_rent, initial_state_rent_matrix, initial_state_capital_land, initial_state_average_income, initial_state_limit_city = compute_equilibrium(fraction_capital_destroyed, amenities, param, housing_limit, population, households_per_income_class, total_RDP, coeff_land, income_net_of_commuting_costs, grid, options, agricultural_rent, interest_rate, number_properties_RDP, average_income, mean_income, income_class_by_housing_type, minimum_housing_supply, param["coeff_A"])

index_min = np.argmin(metrics)
metrics[index_min]
param["pockets"] = save_param_informal_settlements[index_min]
param["pockets"][param["pockets"] < 0.05] = 0.05
param["pockets"][param["pockets"] > 0.99] = 0.99 
param["backyard_pockets"] = save_param_backyards[index_min]
param["backyard_pockets"][param["backyard_pockets"] < 0.05] = 0.05
param["backyard_pockets"][param["backyard_pockets"] > 0.99] = 0.99

os.mkdir('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name)
np.save('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/param_pockets.npy', save_param_informal_settlements[index_min])
np.save('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/param_backyards.npy', save_param_backyards[index_min])

# %% Compute initial state

if options["pluvial"] == 0:
    param["pockets"] = np.load('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/disamenity_parameters_fluvial_IS.npy')
    param["backyard_pockets"] = np.load('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/disamenity_parameters_fluvial_IB.npy')

print("\n*** Solver initial state ***\n")
initial_state_utility, initial_state_error, initial_state_simulated_jobs, initial_state_households_housing_types, initial_state_household_centers, initial_state_households, initial_state_dwelling_size, initial_state_housing_supply, initial_state_rent, initial_state_rent_matrix, initial_state_capital_land, initial_state_average_income, initial_state_limit_city = compute_equilibrium(fraction_capital_destroyed, amenities, param, housing_limit, population, households_per_income_class, total_RDP, coeff_land, income_net_of_commuting_costs, grid, options, agricultural_rent, interest_rate, number_properties_RDP, average_income, mean_income, income_class_by_housing_type, minimum_housing_supply, param["coeff_A"])

# %% Validation

if options["agents_anticipate_floods"] == 0:
    fraction_capital_destroyed = import_floods_data()

#General validation
export_housing_types(initial_state_households_housing_types, initial_state_household_centers, housing_type_data, households_per_income_class, name, 'Simulation', 'Data')
#export_density_rents_sizes(grid, name, data_rdp, housing_types_grid, initial_state_households_housing_types, initial_state_dwelling_size, initial_state_rent, simul1_households_housing_type, simul1_rent, simul1_dwelling_size, data_sp["dwelling_size"], SP_code)
validation_density(grid, initial_state_households_housing_types, name, housing_types)
validation_density_housing_types(grid,initial_state_households_housing_types, housing_types, name, 0)
validation_housing_price(grid, initial_state_rent, interest_rate, param, center, name)
#plot_diagnosis_map_informl(grid, coeff_land, initial_state_households_housing_types, name)

#Stats per housing type (flood depth and households in flood-prone areas)
fluvial_floods = ['FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr', 'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr', 'FD_1000yr']
pluvial_floods = ['P_5yr', 'P_10yr', 'P_20yr', 'P_50yr', 'P_75yr', 'P_100yr', 'P_200yr', 'P_250yr', 'P_500yr', 'P_1000yr']
path_data = "C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/"
count_formal = housing_types.formal_grid - (number_properties_RDP * total_RDP / sum(number_properties_RDP))
count_formal[count_formal < 0] = 0
stats_per_housing_type_data_fluvial = compute_stats_per_housing_type(fluvial_floods, path_data, count_formal, (number_properties_RDP * total_RDP / sum(number_properties_RDP)), housing_types.informal_grid, housing_types.backyard_formal_grid + housing_types.backyard_informal_grid, options, param)
stats_per_housing_type_fluvial = compute_stats_per_housing_type(fluvial_floods, path_data, initial_state_households_housing_types[0, :], initial_state_households_housing_types[3, :], initial_state_households_housing_types[2, :], initial_state_households_housing_types[1, :], options, param)
validation_flood(name, stats_per_housing_type_data_fluvial, stats_per_housing_type_fluvial, 'Data', 'Simul', 'fluvial')
if options["pluvial"] == 1:
    stats_per_housing_type_data_pluvial = compute_stats_per_housing_type(pluvial_floods, path_data, count_formal, (number_properties_RDP * total_RDP / sum(number_properties_RDP)), housing_types.informal_grid, housing_types.backyard_formal_grid + housing_types.backyard_informal_grid, options, param)
    stats_per_housing_type_pluvial = compute_stats_per_housing_type(pluvial_floods, path_data, initial_state_households_housing_types[0, :], initial_state_households_housing_types[3, :], initial_state_households_housing_types[2, :], initial_state_households_housing_types[1, :], options, param)
    validation_flood(name, stats_per_housing_type_data_pluvial, stats_per_housing_type_pluvial, 'Data', 'Simul', 'pluvial')


#Damages
formal_structure_cost_data = compute_formal_structure_cost_method1(data_sp["price"], data_sp["dwelling_size"], data_sp["sp_code"], grid)
content_cost_data = pd.DataFrame(data = {'formal': np.empty(24014), 'informal': np.empty(24014), 'subsidized': np.empty(24014), 'backyard': np.empty(24014)})
content_cost_data[:] = np.nan
damages_data_fluvial = compute_damages(fluvial_floods, path_data, param, content_cost_data,
                    count_formal, (number_properties_RDP * total_RDP / sum(number_properties_RDP)), housing_types.informal_grid, housing_types.backyard_formal_grid + housing_types.backyard_informal_grid,
                    formal_structure_cost_data, content_damages, structural_damages_type4b, structural_damages_type2, structural_damages_type3a, options)
    
formal_structure_cost_simul = compute_formal_structure_cost_method2(initial_state_rent, param, interest_rate, coeff_land, initial_state_households_housing_types)    
content_cost_simul = compute_content_cost(initial_state_household_centers, income_net_of_commuting_costs, param, fraction_capital_destroyed, initial_state_rent, initial_state_dwelling_size, interest_rate)
damages_simul_fluvial = compute_damages(fluvial_floods, path_data, param, content_cost_simul,
                    initial_state_households_housing_types[0, :], initial_state_households_housing_types[3, :], initial_state_households_housing_types[2, :], initial_state_households_housing_types[1, :],
                    formal_structure_cost_simul, content_damages, structural_damages_type4b, structural_damages_type2, structural_damages_type3a, options)
    
plot_damages(damages_simul_fluvial, name)
compare_damages(damages_simul_fluvial, damages_data_fluvial, "Simulation", "Data", name)

if options["pluvial"] == 1:
    damages_data_pluvial = compute_damages(pluvial_floods, path_data, param, content_cost_data,
                    count_formal, (number_properties_RDP * total_RDP / sum(number_properties_RDP)), housing_types.informal_grid, housing_types.backyard_formal_grid + housing_types.backyard_informal_grid,
                    formal_structure_cost_data, content_damages, structural_damages_type4b, structural_damages_type2, structural_damages_type3a, options)
    
    damages_simul_pluvial = compute_damages(pluvial_floods, path_data, param, content_cost_simul,
                    initial_state_households_housing_types[0, :], initial_state_households_housing_types[3, :], initial_state_households_housing_types[2, :], initial_state_households_housing_types[1, :],
                    formal_structure_cost_simul, content_damages, structural_damages_type4b, structural_damages_type2, structural_damages_type3a, options)
    
    plot_damages(damages_simul_pluvial, name)
    compare_damages(damages_simul_pluvial, damages_data_pluvial, "Simulation", "Data", name)

# %% Scenarios

#Compute scenarios
t = np.arange(0, 29)
param["informal_structure_value_ref"] = copy.deepcopy(param["informal_structure_value"])
param["subsidized_structure_value_ref"] = copy.deepcopy(param["subsidized_structure_value"])
simulation_households_center, simulation_households_housing_type, simulation_dwelling_size, simulation_rent, simulation_households, simulation_error, simulation_housing_supply, simulation_utility, simulation_deriv_housing, simulation_T = run_simulation(t, options, income_2011, param, grid, initial_state_utility, initial_state_error, initial_state_households, initial_state_households_housing_types, initial_state_housing_supply, initial_state_household_centers, initial_state_average_income, initial_state_rent, initial_state_dwelling_size, fraction_capital_destroyed, amenities, housing_limit, spline_estimate_RDP, spline_land_constraints, spline_land_backyard, spline_land_RDP, spline_land_informal, income_class_by_housing_type)

#Export outputs
name = name + "_scenarios"
os.mkdir('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name)
export_housing_types(simulation_households_housing_type[0, :, :], simulation_households_center[0, :, :], np.nansum(simulation_households_housing_type[29, :, :], 1), np.nansum(simulation_households_center[29, :, :], 1), name, '2011', '2040')

data = pd.DataFrame({'2011': np.nansum(simulation_households_housing_type[0, :, :], 1), '2020': np.nansum(simulation_households_housing_type[9, :, :], 1),'2030': np.nansum(simulation_households_housing_type[19, :, :], 1),'2040': np.nansum(simulation_households_housing_type[29, :, :], 1),}, index = ["Formal private", "Informal in \n backyards", "Informal \n settlements", "Formal subsidized"])
data.plot(kind="bar")
plt.tick_params(labelbottom=True)
plt.xticks(rotation='horizontal')
    
simul_2011 = simulation_households_housing_type[0, :, :]
simul_2020 = simulation_households_housing_type[9, :, :]
simul_2030 = simulation_households_housing_type[19, :, :]
simul_2040 = simulation_households_housing_type[29, :, :]

np.save('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/housing_types_2011.npy', simul_2011)
np.save('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/housing_types_2020.npy', simul_2020)
np.save('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/housing_types_2030.npy', simul_2030)
np.save('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/housing_types_2040.npy', simul_2040)


housing_types_2011 = np.load('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/fluvial/test/housing_types_2011.npy')
housing_types_2020 = np.load('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/fluvial/test/housing_types_2020.npy')
housing_types_2030 = np.load('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/fluvial/test/housing_types_2030.npy')
housing_types_2040 = np.load('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/fluvial/test/housing_types_2040.npy')


#### 1. Exposition des pop vulnérables

##  A. Evolution des prop flood prone areas

### !!! A faire : rajouter un seuil !!!!

stats_per_housing_type_2011_fluvial = compute_stats_per_housing_type(fluvial_floods, path_data, simulation_households_housing_type[0, 0, :], simulation_households_housing_type[0, 3, :], simulation_households_housing_type[0, 2, :], simulation_households_housing_type[0, 1, :], options, param)
stats_per_housing_type_2020_fluvial = compute_stats_per_housing_type(fluvial_floods, path_data, simulation_households_housing_type[9, 0, :], simulation_households_housing_type[9, 3, :], simulation_households_housing_type[9, 2, :], simulation_households_housing_type[9, 1, :], options, param)
stats_per_housing_type_2030_fluvial = compute_stats_per_housing_type(fluvial_floods, path_data, simulation_households_housing_type[19, 0, :], simulation_households_housing_type[19, 3, :], simulation_households_housing_type[19, 2, :], simulation_households_housing_type[19, 1, :], options, param)
stats_per_housing_type_2040_fluvial = compute_stats_per_housing_type(fluvial_floods, path_data, simulation_households_housing_type[28, 0, :], simulation_households_housing_type[28, 3, :], simulation_households_housing_type[28, 2, :], simulation_households_housing_type[28, 1, :], options, param)

label = ["Formal private", "Formal subsidized", "Informal \n settlements", "Informal \n in backyards"]
stats_2011_1 = [stats_per_housing_type_2011_fluvial.fraction_formal_in_flood_prone_area[2], stats_per_housing_type_2011_fluvial.fraction_subsidized_in_flood_prone_area[2], stats_per_housing_type_2011_fluvial.fraction_informal_in_flood_prone_area[2], stats_per_housing_type_2011_fluvial.fraction_backyard_in_flood_prone_area[2]]
stats_2011_2 = [stats_per_housing_type_2011_fluvial.fraction_formal_in_flood_prone_area[3], stats_per_housing_type_2011_fluvial.fraction_subsidized_in_flood_prone_area[3], stats_per_housing_type_2011_fluvial.fraction_informal_in_flood_prone_area[3], stats_per_housing_type_2011_fluvial.fraction_backyard_in_flood_prone_area[3]]
stats_2011_3 = [stats_per_housing_type_2011_fluvial.fraction_formal_in_flood_prone_area[5], stats_per_housing_type_2011_fluvial.fraction_subsidized_in_flood_prone_area[5], stats_per_housing_type_2011_fluvial.fraction_informal_in_flood_prone_area[5], stats_per_housing_type_2011_fluvial.fraction_backyard_in_flood_prone_area[5]]
stats_2040_1 = [stats_per_housing_type_2040_fluvial.fraction_formal_in_flood_prone_area[2], stats_per_housing_type_2040_fluvial.fraction_subsidized_in_flood_prone_area[2], stats_per_housing_type_2040_fluvial.fraction_informal_in_flood_prone_area[2], stats_per_housing_type_2040_fluvial.fraction_backyard_in_flood_prone_area[2]]
stats_2040_2 = [stats_per_housing_type_2040_fluvial.fraction_formal_in_flood_prone_area[3], stats_per_housing_type_2040_fluvial.fraction_subsidized_in_flood_prone_area[3], stats_per_housing_type_2040_fluvial.fraction_informal_in_flood_prone_area[3], stats_per_housing_type_2040_fluvial.fraction_backyard_in_flood_prone_area[3]]
stats_2040_3 = [stats_per_housing_type_2040_fluvial.fraction_formal_in_flood_prone_area[5], stats_per_housing_type_2040_fluvial.fraction_subsidized_in_flood_prone_area[5], stats_per_housing_type_2040_fluvial.fraction_informal_in_flood_prone_area[5], stats_per_housing_type_2040_fluvial.fraction_backyard_in_flood_prone_area[5]]
colors = ['#FF9999', '#00BFFF','#C1FFC1','#CAE1FF','#FFDEAD']
r = np.arange(4)
barWidth = 0.25
plt.figure(figsize=(10,7))
plt.bar(r, stats_2011_1, color=colors[0], edgecolor='white', width=barWidth, label="20 years")
plt.bar(r, np.array(stats_2011_2) - np.array(stats_2011_1), bottom=np.array(stats_2011_1), color=colors[1], edgecolor='white', width=barWidth, label='50 years')
plt.bar(r, np.array(stats_2011_3) - (np.array(stats_2011_2)), bottom=(np.array(stats_2011_2)), color=colors[2], edgecolor='white', width=barWidth, label='100 years')
plt.bar(r + 0.25, np.array(stats_2040_1), color=colors[0], edgecolor='white', width=barWidth)
plt.bar(r + 0.25, np.array(stats_2040_2) - np.array(stats_2040_1), bottom=np.array(stats_2040_1), color=colors[1], edgecolor='white', width=barWidth)
plt.bar(r + 0.25, np.array(stats_2040_3) - np.array(stats_2040_2), bottom=np.array(stats_2040_2), color=colors[2], edgecolor='white', width=barWidth)
plt.legend(loc = 'upper right')
plt.xticks(r, label)
plt.text(r[0] - 0.1, stats_per_housing_type_2011_fluvial.fraction_formal_in_flood_prone_area[5] + 0.005, "2011")
plt.text(r[1] - 0.1, stats_per_housing_type_2011_fluvial.fraction_subsidized_in_flood_prone_area[5] + 0.005, "2011") 
plt.text(r[2] - 0.1, stats_per_housing_type_2011_fluvial.fraction_informal_in_flood_prone_area[5] + 0.005, "2011") 
plt.text(r[3] - 0.1, stats_per_housing_type_2011_fluvial.fraction_backyard_in_flood_prone_area[5] + 0.005, "2011")
plt.text(r[0] + 0.15, stats_per_housing_type_2040_fluvial.fraction_formal_in_flood_prone_area[5] + 0.005, '2040')
plt.text(r[1] + 0.15, stats_per_housing_type_2040_fluvial.fraction_subsidized_in_flood_prone_area[5] + 0.005, '2040') 
plt.text(r[2] + 0.15, stats_per_housing_type_2040_fluvial.fraction_informal_in_flood_prone_area[5] + 0.005, '2040') 
plt.text(r[3] + 0.15, stats_per_housing_type_2040_fluvial.fraction_backyard_in_flood_prone_area[5] + 0.005, '2040') 
plt.tick_params(labelbottom=True)
plt.ylabel("Dwellings in flood-prone areas (%)")
plt.show()

##  B. Evolution des dégâts par quartile de revenu

item = 'FD_100yr'
path_data = "C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/"
option = "percent" #"absolu"
df = pd.DataFrame()
type_flood = copy.deepcopy(item)
data_flood = np.squeeze(pd.read_excel(path_data + item + ".xlsx"))
        
df['formal_structure_damages'] = formal_structure_cost * depth_damage_function_structure(data_flood['flood_depth'])
df['subsidized_structure_damages'] = param["subsidized_structure_value"] * depth_damage_function_structure(data_flood['flood_depth'])
df['informal_structure_damages'] = param["informal_structure_value"] * depth_damage_function_structure(data_flood['flood_depth'])
df['backyard_structure_damages'] = param["informal_structure_value"] * depth_damage_function_structure(data_flood['flood_depth'])
        
df['formal_content_damages'] =  content_cost.formal * depth_damage_function_contents(data_flood['flood_depth'])
df['subsidized_content_damages'] = content_cost.subsidized * depth_damage_function_contents(data_flood['flood_depth'])
df['informal_content_damages'] = content_cost.informal * depth_damage_function_contents(data_flood['flood_depth'])
df['backyard_content_damages'] = content_cost.backyard * depth_damage_function_contents(data_flood['flood_depth'])

df["prop_flood_prone"] = data_flood["prop_flood_prone"]

#peut-être à faire en version 2011/2040
#peut-être à faire pour les différentes classes de revenus
plot_damages_per_hh('formal', name, simulation_households_housing_type[28, 0, :], simulation_households[28, 0, :], df)

#formel
df_pop_flood_prone_formel = copy.deepcopy(df.prop_flood_prone) * simulation_households_housing_type[0, 0, :]
df_pop_flood_prone_backyard = copy.deepcopy(df.prop_flood_prone) * simulation_households_housing_type[0, 1, :]
df_pop_flood_prone_informal = copy.deepcopy(df.prop_flood_prone) * simulation_households_housing_type[0, 2, :]
df_pop_flood_prone_subsidized = copy.deepcopy(df.prop_flood_prone) * simulation_households_housing_type[0, 3, :]
df["total_damages_formal"] = df['formal_structure_damages'] + df['formal_content_damages']
df["total_damages_backyard"] = df['backyard_structure_damages'] + df['backyard_content_damages']
df["total_damages_informal"] = df['informal_structure_damages'] + df['informal_content_damages']
df["total_damages_subsidized"] = df['subsidized_structure_damages'] + df['subsidized_content_damages']

df["total_damages_formal"][np.isnan(df["total_damages_formal"])] = 0
df["total_damages_backyard"][np.isnan(df["total_damages_backyard"])] = 0
df["total_damages_informal"][np.isnan(df["total_damages_informal"])] = 0
df["total_damages_subsidized"][np.isnan(df["total_damages_subsidized"])] = 0

sum(df["total_damages_formal"] * df_pop_flood_prone_formel) / sum(simulation_households_housing_type[0, 0, :])
sum(df["total_damages_backyard"] * df_pop_flood_prone_backyard) / sum(simulation_households_housing_type[0, 1, :])
sum(df["total_damages_informal"] * df_pop_flood_prone_informal) / sum(simulation_households_housing_type[0, 2, :])
sum(df["total_damages_subsidized"] * df_pop_flood_prone_subsidized) / sum(simulation_households_housing_type[0, 3, :])

np.average(df["total_damages_formal"], weights = df_pop_flood_prone_formel)
np.average(df["total_damages_informal"], weights = df_pop_flood_prone_informal)
np.average(df["total_damages_backyard"], weights = df_pop_flood_prone_backyard)
np.average(df["total_damages_subsidized"], weights = df_pop_flood_prone_subsidized)

households_2011 = simulation_households[0, :,:,:]

    
######### 2. Decomposition

spline_agricultural_rent, spline_interest_rate, spline_RDP, spline_population_income_distribution, spline_inflation, spline_income_distribution, spline_population, spline_interest_rate, spline_income, spline_minimum_housing_supply, spline_fuel = import_scenarios(income_2011, param, grid)

formal_structure_cost_2011 = compute_formal_structure_cost_method2(simulation_rent[0, :, :], param, interpolate_interest_rate(spline_interest_rate, 0), import_coeff_land(spline_land_constraints, spline_land_backyard, spline_land_informal, spline_land_RDP, param, 0), simulation_households_housing_type[0, :, :], (spline_income(0) / param["income_year_reference"])**(-param["coeff_b"]) * param["coeff_A"])    
content_cost_2011 = compute_content_cost(simulation_households_center[0, :, :], np.load("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/precalculated_transport/year_0.npy"), param, fraction_capital_destroyed, simulation_rent[0, :, :], simulation_dwelling_size[0, :, :], interpolate_interest_rate(spline_interest_rate, 0))
damages_fluvial_2011 = compute_damages(fluvial_floods, path_data, param, content_cost_simul,
                    simulation_households_housing_type[0, 0, :], simulation_households_housing_type[0, 3, :], simulation_households_housing_type[0, 2, :], simulation_households_housing_type[0, 1, :],
                    formal_structure_cost_simul, content_damages, structural_damages_type4b, structural_damages_type2, structural_damages_type3a, options)

formal_structure_cost_2040 = compute_formal_structure_cost_method2(simulation_rent[28, :, :], param, interpolate_interest_rate(spline_interest_rate, 29), import_coeff_land(spline_land_constraints, spline_land_backyard, spline_land_informal, spline_land_RDP, param, 29), simulation_households_housing_type[28, :, :], (spline_income(29) / param["income_year_reference"])**(-param["coeff_b"]) * param["coeff_A"])    
content_cost_2040 = compute_content_cost(simulation_households_center[28, :, :], np.load("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/precalculated_transport/year_29.npy"), param, fraction_capital_destroyed, simulation_rent[28, :, :], simulation_dwelling_size[28, :, :], interpolate_interest_rate(spline_interest_rate, 29))
damages_fluvial_2040 = compute_damages(fluvial_floods, path_data, param, content_cost_simul,
                    simulation_households_housing_type[28, 0, :], simulation_households_housing_type[28, 3, :], simulation_households_housing_type[28, 2, :], simulation_households_housing_type[28, 1, :],
                    formal_structure_cost_simul, content_damages, structural_damages_type4b, structural_damages_type2, structural_damages_type3a, options)

damages_fluvial_2011.backyard_damages = damages_fluvial_2011.backyard_content_damages + damages_fluvial_2011.backyard_structure_damages
damages_fluvial_2011.informal_damages = damages_fluvial_2011.informal_content_damages + damages_fluvial_2011.informal_structure_damages
damages_fluvial_2011.subsidized_damages = damages_fluvial_2011.subsidized_content_damages + damages_fluvial_2011.subsidized_structure_damages
damages_fluvial_2011.formal_damages = damages_fluvial_2011.formal_content_damages + damages_fluvial_2011.formal_structure_damages

damages_fluvial_2040.backyard_damages = damages_fluvial_2040.backyard_content_damages + damages_fluvial_2040.backyard_structure_damages
damages_fluvial_2040.informal_damages = damages_fluvial_2040.informal_content_damages + damages_fluvial_2040.informal_structure_damages
damages_fluvial_2040.subsidized_damages = damages_fluvial_2040.subsidized_content_damages + damages_fluvial_2040.subsidized_structure_damages
damages_fluvial_2040.formal_damages = damages_fluvial_2040.formal_content_damages + damages_fluvial_2040.formal_structure_damages


label = ["2011", "2040"]
stats_2011_formal = [annualize_damages(damages_fluvial_2011.formal_damages),annualize_damages(damages_fluvial_2040.formal_damages)]
stats_2011_subsidized = [annualize_damages(damages_fluvial_2011.subsidized_damages),annualize_damages(damages_fluvial_2040.subsidized_damages)]
stats_2011_informal = [annualize_damages(damages_fluvial_2011.informal_damages), annualize_damages(damages_fluvial_2040.informal_damages)]
stats_2011_backyard = [annualize_damages(damages_fluvial_2011.backyard_damages),annualize_damages(damages_fluvial_2040.backyard_damages)]
colors = ['#FF9999', '#00BFFF','#C1FFC1','#CAE1FF','#FFDEAD']
r = np.arange(2)
barWidth = 0.25
plt.figure(figsize=(10,7))
plt.bar(r, stats_2011_formal, color=colors[0], edgecolor='white', width=barWidth, label="formal")
plt.bar(r, np.array(stats_2011_subsidized), bottom=np.array(stats_2011_formal), color=colors[1], edgecolor='white', width=barWidth, label='subsidized')
plt.bar(r, np.array(stats_2011_informal), bottom=(np.array(stats_2011_subsidized) + np.array(stats_2011_formal)), color=colors[2], edgecolor='white', width=barWidth, label='informal')
plt.bar(r, np.array(stats_2011_backyard), bottom=(np.array(stats_2011_informal) + np.array(stats_2011_subsidized) + np.array(stats_2011_formal)), color=colors[3], edgecolor='white', width=barWidth, label='backyard')
plt.legend()
plt.xticks(r, label)
plt.tick_params(labelbottom=True)
plt.ylabel("Estimated annual damages (R)")
plt.show()

######### 3. Climate change















