# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:32:48 2020

@author: Charlotte Liotta
"""

import scipy
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import copy

from data import *

def error_map(error, grid, export_name):
    map = plt.scatter(grid.x, 
            grid.y, 
            s=None,
            c=error,
            cmap = 'RdYlGn',
            marker='.')
    plt.colorbar(map)
    plt.axis('off')
    plt.clim(-100, 100)
    plt.savefig(export_name)
    plt.close()
    
def export_map(value, grid, export_name, lim):
    map = plt.scatter(grid.x, 
            grid.y, 
            s=None,
            c=value,
            cmap = 'Reds',
            marker='.')
    plt.colorbar(map)
    plt.axis('off')
    plt.clim(0, lim)
    plt.savefig(export_name)
    plt.close()
    
# %% Validation

def export_housing_types(housing_type_1, households_center_1, housing_type_2, households_center_2, name, legend1, legend2):
    
    #Graph validation housing type
    data = pd.DataFrame({legend1: np.nansum(housing_type_1, 1), legend2: housing_type_2}, index = ["FP", "IB", "IS", "FS"])
    data.plot(kind="bar")
    plt.title("Housing types")
    plt.ylabel("Households")
    plt.tick_params(labelbottom=True)
    plt.savefig('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/validation_housing_type.png')
    plt.close()
    
    #Graph validation income class
    data = pd.DataFrame({legend1: np.nansum(households_center_1, 1), legend2: households_center_2}, index = ["Class 1", "Class 2", "Class 3", "Class 4"])
    data.plot(kind="bar")
    plt.title("Income classes")
    plt.ylabel("Households")
    plt.tick_params(labelbottom=True)
    plt.savefig('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/validation_income_class.png')
    plt.close()
        
def export_density_rents_sizes(grid, name, data_rdp, housing_types_grid, initial_state_households_housing_types, initial_state_dwelling_size, initial_state_rent, simul1_households_housing_type, simul1_rent, simul1_dwelling_size, dwelling_size_sp, SP_code):

    #1. Housing types

    count_formal = housing_types_grid.formal_grid_2011 - data_rdp["count"]
    count_formal[count_formal < 0] = 0

    os.mkdir('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/housing_types')

    #Formal
    error = (initial_state_households_housing_types[0, :] / count_formal - 1) * 100
    error_map(error, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/formal_diff_with_data.png')  
    export_map(count_formal, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/formal_data.png', 1200)
    export_map(initial_state_households_housing_types[0, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/formal_simul.png', 1200)
    export_map(simul1_households_housing_type[0, 0, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/formal_Basile1.png', 1200)
    
    #Subsidized
    error = (initial_state_households_housing_types[3, :] / data_rdp["count"] - 1) * 100
    error_map(error, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/subsidized_diff_with_data.png')  
    export_map(data_rdp["count"], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/subsidized_data.png', 1200)
    export_map(initial_state_households_housing_types[3, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/subsidized_simul.png', 1200)
    export_map(simul1_households_housing_type[0, 3, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/subsidized_Basile1.png', 1200)
    
    #Informal
    error = (initial_state_households_housing_types[2, :] / housing_types_grid.informal_grid_2011 - 1) * 100
    error_map(error, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/informal_diff_with_data.png')  
    export_map(housing_types_grid.informal_grid_2011, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/informal_data.png', 800)
    export_map(initial_state_households_housing_types[2, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/informal_simul.png', 800)
    export_map(simul1_households_housing_type[0, 2, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/informal_Basile1.png', 800)
    
    #Backyard
    error = (initial_state_households_housing_types[1, :] / housing_types_grid.backyard_grid_2011 - 1) * 100
    error_map(error, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/backyard_diff_with_data.png')  
    export_map(housing_types_grid.backyard_grid_2011, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/backyard_data.png', 800)
    export_map(initial_state_households_housing_types[1, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/housing_types/backyard_simul.png', 800)
    export_map(simul1_households_housing_type[0, 1, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/backyard_Basile1.png', 800)
    
    #2. Dwelling size
    
    os.mkdir('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size')
    
    dwelling_size = SP_to_grid_2011_1(dwelling_size_sp, SP_code, grid)
    
    #Data
    export_map(dwelling_size, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/data.png', 300)
    
    #Class 1
    error = (initial_state_dwelling_size[0, :] / dwelling_size - 1) * 100
    error_map(error, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class1_diff_with_data.png')  
    export_map(initial_state_dwelling_size[0, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class1_simul.png', 300)
    export_map(simul1_dwelling_size[0, 0, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class1_Basile1.png', 300)
    
    #Class 2
    error = (initial_state_dwelling_size[1, :] / dwelling_size - 1) * 100
    error_map(error, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class2_diff_with_data.png')  
    export_map(initial_state_dwelling_size[1, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class2_simul.png', 200)
    export_map(simul1_dwelling_size[0, 1, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class2_Basile1.png', 200)
    
    #Class 3
    error = (initial_state_dwelling_size[2, :] / dwelling_size - 1) * 100
    error_map(error, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class3_diff_with_data.png')  
    export_map(initial_state_dwelling_size[2, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class3_simul.png', 200)
    export_map(simul1_dwelling_size[0, 2, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class3_Basile1.png', 200)
    
    #Class 4
    error = (initial_state_dwelling_size[3, :] / dwelling_size - 1) * 100
    error_map(error, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class4_diff_with_data.png')  
    export_map(initial_state_dwelling_size[3, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class4_simul.png', 100)
    export_map(simul1_dwelling_size[0, 3, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class4_Basile1.png', 100)
    
    #3. Rents

    os.mkdir('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents')
    
    #Class 1
    export_map(initial_state_rent[0, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents/class1_simul.png', 800)
    export_map(simul1_rent[0, 0, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents/class1_Basile1.png', 800)
    
    #Class 2
    export_map(initial_state_rent[1, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents/class2_simul.png', 700)
    export_map(simul1_rent[0, 1, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents/class2_Basile1.png', 700)
    
    #Class 3
    export_map(initial_state_rent[2, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents/class3_simul.png', 600)
    export_map(simul1_rent[0, 2, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents/class3_Basile1.png', 600)
    
    #Class 4
    export_map(initial_state_rent[3, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents/class4_simul.png', 500)
    export_map(simul1_rent[0, 3, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents/class4_Basile1.png', 500)

def validation_density(grid, initial_state_households_housing_types, name):
    
    #Population density
    xData = grid.dist
    data = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/0. Precalculated inputs/data.mat')['data']
    nb_poor = data["gridNumberPoor"][0][0].squeeze()
    nb_rich = data["gridNumberRich"][0][0].squeeze()
    yData = (nb_poor + nb_rich) / 0.25
    ySimul = np.nansum(initial_state_households_housing_types, 0) / 0.25

    df = pd.DataFrame(data = np.transpose(np.array([xData, yData, ySimul])), columns = ["x", "yData", "ySimul"])
    df["round"] = round(df.x)
    new_df = df.groupby(['round']).mean()
    q1_df = df.groupby(['round']).quantile(0.25)
    q3_df = df.groupby(['round']).quantile(0.75)

    plt.plot(np.arange(max(df["round"] + 1)), new_df.yData, color = "black", label = "Data")
    plt.plot(np.arange(max(df["round"] + 1)), new_df.ySimul, color = "green", label = "Simulation")
    axes = plt.axes()
    axes.set_ylim([0, 2000])
    axes.set_xlim([0, 50])
    axes.fill_between(np.arange(max(df["round"] + 1)), q1_df.ySimul, q3_df.ySimul, color = "lightgreen")
    plt.legend()
    plt.xlabel("Distance to the city center (km)")
    plt.ylabel("Households density (per km2)") 
    plt.tick_params(bottom = True,labelbottom=True)   
    plt.title("Population density")
    plt.savefig('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/validation_density.png')  
    plt.close()
    
def validation_density_housing_types(grid,initial_state_households_housing_types, housing_types_grid, name):

    #Housing types
    xData = grid.dist
    formal_data = (housing_types_grid.formal_grid_2011) / 0.25
    backyard_data = (housing_types_grid.backyard_grid_2011) / 0.25
    informal_data = (housing_types_grid.informal_grid_2011) / 0.25
    formal_simul = (initial_state_households_housing_types[0, :] + initial_state_households_housing_types[3, :]) / 0.25
    informal_simul = (initial_state_households_housing_types[2, :]) / 0.25
    backyard_simul = (initial_state_households_housing_types[1, :]) / 0.25

    df = pd.DataFrame(data = np.transpose(np.array([xData, formal_data, backyard_data, informal_data, formal_simul, backyard_simul, informal_simul])), columns = ["xData", "formal_data", "backyard_data", "informal_data", "formal_simul", "backyard_simul", "informal_simul"])
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).mean()

    plt.plot(np.arange(max(df["round"] + 1)), new_df.formal_data, color = "black", label = "Data")
    plt.plot(np.arange(max(df["round"] + 1)), new_df.formal_simul, color = "green", label = "Simulation")
    axes = plt.axes()
    axes.set_ylim([0, 1600])
    axes.set_xlim([0, 40])
    #plt.title("Formal")
    plt.legend()
    plt.xlabel("Distance to the city center (km)")
    plt.ylabel("Households density (per km2)")
    plt.savefig('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/validation_density_formal.png')  
    plt.close()

    plt.plot(np.arange(max(df["round"] + 1)), new_df.informal_data, color = "black", label = "Data")
    plt.plot(np.arange(max(df["round"] + 1)), new_df.informal_simul, color = "green", label = "Simulation")
    axes = plt.axes()
    axes.set_ylim([0, 400])
    axes.set_xlim([0, 40])
    #plt.title("Informal")
    plt.xlabel("Distance to the city center (km)")
    plt.ylabel("Households density (per km2)")
    plt.legend()
    plt.savefig('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/validation_density_informal.png')  
    plt.close()

    plt.plot(np.arange(max(df["round"] + 1)), new_df.backyard_data, color = "black", label = "Data")
    plt.plot(np.arange(max(df["round"] + 1)), new_df.backyard_simul, color = "green", label = "Simulation")
    axes = plt.axes()
    axes.set_ylim([0, 450])
    axes.set_xlim([0, 40])
    #plt.title("Backyard")
    plt.legend()
    plt.xlabel("Distance to the city center (km)")
    plt.ylabel("Households density (per km2)")
    plt.savefig('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/validation_density_backyard.png')  
    plt.close()

def validation_housing_price(grid, initial_state_rent, interest_rate, param, center, name):
    
    data = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/0. Precalculated inputs/data.mat')['data']
    sp_x = data['spX'][0][0].squeeze() #Number of informal settlements in backyard per grid cell, 24014
    sp_y = data['spY'][0][0].squeeze()
    sp_price = data['spPrice'][0][0].squeeze()[2, :]

    priceSimul = (initial_state_rent[0, :] * param["coeff_A"] * param["coeff_b"] ** param["coeff_b"] / (interest_rate + param["depreciation_rate"])) ** (1/param["coeff_a"])
    priceSimulPricePoints = griddata(np.transpose(np.array([grid.x, grid.y])), priceSimul, np.transpose(np.array([sp_x, sp_y])))

    xData = np.sqrt((sp_x - center[0]) ** 2 + (sp_y - center[1])** 2)
    yData = sp_price
    xSimulation = xData
    ySimulation = priceSimulPricePoints

    df = pd.DataFrame(data = np.transpose(np.array([xData, yData, ySimulation])), columns = ["xData", "yData", "ySimulation"])
    df["round"] = round(df.xData)
    new_df = df.groupby(['round']).mean()

    which = ~np.isnan(new_df.yData) & ~np.isnan(new_df.ySimulation)

    plt.plot(new_df.xData[which], new_df.yData[which], color = "black", label = "Data")
    plt.plot(new_df.xData[which], new_df.ySimulation[which], color = "green", label = "Simul")
    axes = plt.axes()
    plt.xlabel("Distance to the city center (km)")
    plt.ylabel("Price (R/m2 of land)")
    plt.legend()
    plt.tick_params(bottom = True,labelbottom=True)
    plt.savefig('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/validation_housing_price.png')  
    plt.close()
    



