# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:58:33 2020

@author: Charlotte Liotta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from flood_outputs import *
from plot_damages_per_household import *

#0. Flood damages per household

#formal_structure_cost depth_damage_function_structure depth_damage_function_contents name param initial_state_households_housing_types

floods = ['FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr', 'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr', 'FD_1000yr']
path_data = "C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/"

option = "percent" #"absolu"

for item in floods:
    
    df = pd.DataFrame()
    
    type_flood = copy.deepcopy(item)
    data_flood = np.squeeze(pd.read_excel(path_data + item + ".xlsx"))
        
    formal_structure_damages = formal_structure_cost * depth_damage_function_structure(data_flood['flood_depth'])
    subsidized_structure_damages = param["subsidized_structure_value"] * depth_damage_function_structure(data_flood['flood_depth'])
    informal_structure_damages = param["informal_structure_value"] * depth_damage_function_structure(data_flood['flood_depth'])
    backyard_structure_damages = param["informal_structure_value"] * depth_damage_function_structure(data_flood['flood_depth'])
        
    formal_content_damages =  content_cost.formal * depth_damage_function_contents(data_flood['flood_depth'])
    subsidized_content_damages = content_cost.subsidized * depth_damage_function_contents(data_flood['flood_depth'])
    informal_content_damages = content_cost.informal * depth_damage_function_contents(data_flood['flood_depth'])
    backyard_content_damages = content_cost.backyard * depth_damage_function_contents(data_flood['flood_depth'])
        
    df['formal_structure_damages'] = formal_structure_damages
    df['subsidized_structure_damages'] = subsidized_structure_damages
    df['informal_structure_damages'] = informal_structure_damages
    df['backyard_structure_damages'] = backyard_structure_damages
    df['formal_content_damages'] = formal_content_damages
    df['informal_content_damages'] = informal_content_damages
    df['backyard_content_damages'] = backyard_content_damages
    df['subsidized_content_damages'] = subsidized_content_damages
    df["prop_flood_prone"] = data_flood["prop_flood_prone"]
    writer = pd.ExcelWriter('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + str(item) + '.xlsx')
    df.to_excel(excel_writer = writer)
    writer.save()
   
#1. Option 1 : on veut les damages annualisés

#On annualise les dégâts pour les 4 types de logements
plot_annualize_damages_per_hh('formal', name, initial_state_households_housing_types[0, :], initial_state_households[0,:,:])
plot_annualize_damages_per_hh('backyard', name, initial_state_households_housing_types[1, :], initial_state_households[1,:,:])
plot_annualize_damages_per_hh('informal', name, initial_state_households_housing_types[2, :], initial_state_households[2,:,:])
plot_annualize_damages_per_hh('subsidized', name, initial_state_households_housing_types[3, :], initial_state_households[3,:,:])

def plot_annualize_damages_per_hh(type_housing, name, nb_hh, households):
    
    damages_5yr = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_5yr' + '.xlsx')  
    damages_10yr = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_10yr' + '.xlsx')  
    damages_20yr = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_20yr' + '.xlsx')  
    damages_50yr = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_50yr' + '.xlsx')  
    damages_75yr = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_75yr' + '.xlsx')  
    damages_100yr = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_100yr' + '.xlsx')  
    damages_200yr = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_200yr' + '.xlsx')  
    damages_250yr = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_250yr' + '.xlsx')  
    damages_500yr = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_500yr' + '.xlsx')  
    damages_1000yr = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_1000yr' + '.xlsx')  

    #Calcul de la proportion de la population qui est flood-prone
    damages_1000yr.prop_flood_prone = damages_1000yr.prop_flood_prone - damages_500yr.prop_flood_prone
    damages_500yr.prop_flood_prone = damages_500yr.prop_flood_prone - damages_250yr.prop_flood_prone
    damages_250yr.prop_flood_prone = damages_250yr.prop_flood_prone - damages_200yr.prop_flood_prone
    damages_200yr.prop_flood_prone = damages_200yr.prop_flood_prone - damages_100yr.prop_flood_prone
    damages_100yr.prop_flood_prone = damages_100yr.prop_flood_prone - damages_75yr.prop_flood_prone
    damages_75yr.prop_flood_prone = damages_75yr.prop_flood_prone - damages_50yr.prop_flood_prone
    damages_50yr.prop_flood_prone = damages_50yr.prop_flood_prone - damages_20yr.prop_flood_prone
    damages_20yr.prop_flood_prone = damages_20yr.prop_flood_prone - damages_10yr.prop_flood_prone
    damages_10yr.prop_flood_prone = damages_10yr.prop_flood_prone - damages_5yr.prop_flood_prone

    df_prop_flood_prone = pd.DataFrame(data = {"5_yr": damages_5yr.prop_flood_prone, "10_yr" : damages_10yr.prop_flood_prone, "20_yr" : damages_20yr.prop_flood_prone, "50_yr" : damages_50yr.prop_flood_prone, "75_yr" : damages_75yr.prop_flood_prone, "100_yr" : damages_100yr.prop_flood_prone, "200_yr" : damages_200yr.prop_flood_prone, "250_yr" : damages_250yr.prop_flood_prone, "500_yr" : damages_500yr.prop_flood_prone, "1000_yr" : damages_1000yr.prop_flood_prone})
    df_pop_flood_prone = copy.deepcopy(df_prop_flood_prone) * nb_hh[:, None]

    damages_5yr = annualize_damages([damages_5yr, damages_10yr, damages_20yr, damages_50yr, damages_75yr, damages_100yr, damages_200yr, damages_250yr, damages_500yr, damages_1000yr])
    damages_10yr = annualize_damages([0, damages_10yr, damages_20yr, damages_50yr, damages_75yr, damages_100yr, damages_200yr, damages_250yr, damages_500yr, damages_1000yr])
    damages_20yr = annualize_damages([0, 0, damages_20yr, damages_50yr, damages_75yr, damages_100yr, damages_200yr, damages_250yr, damages_500yr, damages_1000yr])
    damages_50yr = annualize_damages([0, 0, 0, damages_50yr, damages_75yr, damages_100yr, damages_200yr, damages_250yr, damages_500yr, damages_1000yr])
    damages_75yr = annualize_damages([0, 0, 0, 0, damages_75yr, damages_100yr, damages_200yr, damages_250yr, damages_500yr, damages_1000yr])
    damages_100yr = annualize_damages([0, 0, 0, 0, 0, damages_100yr, damages_200yr, damages_250yr, damages_500yr, damages_1000yr])
    damages_200yr = annualize_damages([0, 0, 0, 0, 0, 0, damages_200yr, damages_250yr, damages_500yr, damages_1000yr])
    damages_250yr = annualize_damages([0, 0, 0, 0, 0, 0, 0, damages_250yr, damages_500yr, damages_1000yr])
    damages_500yr = annualize_damages([0, 0, 0, 0, 0, 0, 0, 0, damages_500yr, damages_1000yr])
    damages_1000yr = annualize_damages([0, 0, 0, 0, 0, 0, 0, 0, 0, damages_1000yr])

    #On refait des dataframe par housing type

    #On commence par le total
    damages_5yr["total_damages"] = damages_5yr[type_housing + '_structure_damages'] + damages_5yr[type_housing + '_content_damages']
    damages_10yr["total_damages"] = damages_10yr[type_housing + '_structure_damages'] + damages_10yr[type_housing + '_content_damages']
    damages_20yr["total_damages"] = damages_20yr[type_housing + '_structure_damages'] + damages_20yr[type_housing + '_content_damages']
    damages_50yr["total_damages"] = damages_50yr[type_housing + '_structure_damages'] + damages_50yr[type_housing + '_content_damages']
    damages_75yr["total_damages"] = damages_75yr[type_housing + '_structure_damages'] + damages_75yr[type_housing + '_content_damages']
    damages_100yr["total_damages"] = damages_100yr[type_housing + '_structure_damages'] + damages_100yr[type_housing + '_content_damages']
    damages_200yr["total_damages"] = damages_200yr[type_housing + '_structure_damages'] + damages_200yr[type_housing + '_content_damages']
    damages_250yr["total_damages"] = damages_250yr[type_housing + '_structure_damages'] + damages_250yr[type_housing + '_content_damages']
    damages_500yr["total_damages"] = damages_500yr[type_housing + '_structure_damages'] + damages_500yr[type_housing + '_content_damages']
    damages_1000yr["total_damages"] = damages_1000yr[type_housing + '_structure_damages'] + damages_1000yr[type_housing + '_content_damages']  

    total_5yr = np.transpose([damages_5yr.total_damages, damages_5yr[type_housing + '_structure_damages'], damages_5yr[type_housing + '_content_damages'], df_pop_flood_prone["5_yr"]])
    total_10yr = np.transpose([damages_10yr.total_damages, damages_10yr[type_housing + '_structure_damages'], damages_10yr[type_housing + '_content_damages'], df_pop_flood_prone["10_yr"]])
    total_20yr = np.transpose([damages_20yr.total_damages, damages_20yr[type_housing + '_structure_damages'], damages_20yr[type_housing + '_content_damages'], df_pop_flood_prone["20_yr"]])
    total_50yr = np.transpose([damages_50yr.total_damages, damages_50yr[type_housing + '_structure_damages'], damages_50yr[type_housing + '_content_damages'], df_pop_flood_prone["50_yr"]])
    total_75yr = np.transpose([damages_75yr.total_damages, damages_75yr[type_housing + '_structure_damages'], damages_75yr[type_housing + '_content_damages'], df_pop_flood_prone["75_yr"]])
    total_100yr = np.transpose([damages_100yr.total_damages, damages_100yr[type_housing + '_structure_damages'], damages_100yr[type_housing + '_content_damages'], df_pop_flood_prone["100_yr"]])
    total_200yr = np.transpose([damages_200yr.total_damages, damages_200yr[type_housing + '_structure_damages'], damages_200yr[type_housing + '_content_damages'], df_pop_flood_prone["200_yr"]])
    total_250yr = np.transpose([damages_250yr.total_damages, damages_250yr[type_housing + '_structure_damages'], damages_250yr[type_housing + '_content_damages'], df_pop_flood_prone["250_yr"]])
    total_500yr = np.transpose([damages_500yr.total_damages, damages_500yr[type_housing + '_structure_damages'], damages_500yr[type_housing + '_content_damages'], df_pop_flood_prone["500_yr"]])
    total_1000yr = np.transpose([damages_1000yr.total_damages, damages_1000yr[type_housing + '_structure_damages'], damages_1000yr[type_housing + '_content_damages'], df_pop_flood_prone["1000_yr"]])

    total = np.vstack([total_5yr, total_10yr, total_20yr, total_50yr, total_75yr, total_100yr, total_200yr, total_250yr, total_500yr, total_1000yr])
    
    if option == "percent":
        income_class = np.argmax(households, 0)    
        average_income = np.load("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/precalculated_transport/average_income_year_0.npy")
    
        real_income = np.empty(len(income_class))
        for i in range(0, len(income_class)):
            print(i)
            real_income[i] = average_income[np.array(income_class)[i], i]
        
        real_income = np.matlib.repmat(real_income, 1, 10).squeeze()
        total[:, 0] = (total[:, 0] / real_income) * 100
        total[:, 1] = (total[:, 1] / real_income) * 100
        total[:, 2] = (total[:, 2] / real_income) * 100
    
    #total = total[total[:, 0] > 1, :] #A affiner pour ne prendre que ceux qui sont dans une zone inondable 100yr
    array_percentile = weighted_percentile(total[: ,0], total[:, 3], np.arange(0, 1, 0.25))

    total = pd.DataFrame(total)
    total["quantile"] = "Q0"
    for i in range(1, len(array_percentile)):
        total["quantile"][(total.loc[:,0] < array_percentile[i]) & (total.loc[: ,0] >= array_percentile[i - 1])] = "Q" + str(i)
    total["quantile"][(total.loc[: ,0] >= array_percentile[3])] = "Q4"
    total = total.loc[~np.isnan(total.loc[: ,0]), :]
    total = total.loc[total["quantile"] != "Q0", :]
    total = pd.DataFrame(total)
    total_damages = total.groupby('quantile').apply(lambda total: np.average(total.loc[~np.isnan(total.loc[: ,0]), 0], weights=total.loc[~np.isnan(total.loc[: ,0]) ,3]))
    structure_damages = total.groupby('quantile').apply(lambda total: np.average(total.loc[~np.isnan(total.loc[:, 0]) ,1], weights=total.loc[~np.isnan(total.loc[: ,0]) ,3]))
    contents_damages = total.groupby('quantile').apply(lambda total: np.average(total.loc[~np.isnan(total.loc[:, 0]) ,2], weights=total.loc[~np.isnan(total.loc[: ,0]) ,3]))
    hh_per_quantile = total.groupby('quantile').sum()
    
    order = ['Q1', 'Q2', 'Q3','Q4'] #,'Q5','Q6','Q7','Q8','Q9','Q10']
    colors = ['#FF9999', '#00BFFF','#C1FFC1','#CAE1FF','#FFDEAD']
    r = np.arange(4)
    barWidth = 0.25
    plt.figure(figsize=(10,7))
    plt.bar(r, np.array(pd.DataFrame(structure_damages).loc[order]).squeeze(), color=colors[0], edgecolor='white', width=barWidth, label="Structure")
    plt.bar(r, np.array(pd.DataFrame(contents_damages).loc[order]).squeeze(), bottom=np.array(pd.DataFrame(structure_damages).loc[order]).squeeze(), color=colors[1], edgecolor='white', width=barWidth, label='Contents')
    plt.legend(loc = 'upper left')
    plt.tick_params(labelbottom=True)
    plt.xticks(r, order)
    #plt.title(str(int(sum(total.loc[: ,3]))) + ' households \n ' + str(int(100 * sum(total.loc[: ,3])/sum(nb_hh))) + '% of households in ' + type_housing + ' housing')
    plt.title(str(int(sum(total.loc[: ,3]))) + ' households')
    #plt.savefig('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_per_hh_' + type_housing + '_' + option + '.png')  
    #plt.close()

#2. Option 2: on veut les dégâts par return period

damages_100yr = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_100yr' + '.xlsx')  

plot_damages_per_hh('formal', name, df_prop_flood_prone, initial_state_households_housing_types[0, :], initial_state_households[0,:,:], damages_100yr)
plot_damages_per_hh('backyard', name, df_prop_flood_prone, initial_state_households_housing_types[1, :], initial_state_households[1,:,:], damages_100yr)
plot_damages_per_hh('informal', name, df_prop_flood_prone, initial_state_households_housing_types[2, :], initial_state_households[2,:,:], damages_100yr)
plot_damages_per_hh('subsidized', name, df_prop_flood_prone, initial_state_households_housing_types[3, :], initial_state_households[3,:,:], damages_100yr)


def plot_damages_per_hh(type_housing, name, df_prop_flood_prone, nb_hh, households, damages):

    df_pop_flood_prone = copy.deepcopy(damages.prop_flood_prone) * nb_hh

    #On commence par le total
    damages["total_damages"] = damages[type_housing + '_structure_damages'] + damages[type_housing + '_content_damages']
    
    total = np.transpose([damages.total_damages, damages[type_housing + '_structure_damages'], damages[type_housing + '_content_damages'], df_pop_flood_prone])
    
    if option == "percent":
        income_class = np.argmax(initial_state_households[0,:,:], 0)    
        average_income = np.load("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/precalculated_transport/average_income_year_0.npy")
    
        real_income = np.empty(len(income_class))
        for i in range(0, len(income_class)):
            print(i)
            real_income[i] = average_income[np.array(income_class)[i], i]
        
        total[:, 0] = (total[:, 0] / real_income) * 100
        total[:, 1] = (total[:, 1] / real_income) * 100
        total[:, 2] = (total[:, 2] / real_income) * 100
    
    #total = total[total[:, 0] > 1, :] #A affiner pour ne prendre que ceux qui sont dans une zone inondable 100yr
    array_percentile = weighted_percentile(total[: ,0], total[: ,3], np.arange(0, 1, 0.25))

    total = pd.DataFrame(total)
    total["quantile"] = "Q0"
    for i in range(1, len(array_percentile)):
        total["quantile"][(total.loc[:,0] < array_percentile[i]) & (total.loc[: ,0] >= array_percentile[i - 1])] = "Q" + str(i)
    total["quantile"][(total.loc[: ,0] >= array_percentile[3])] = "Q4"
    total = total.loc[~np.isnan(total.loc[: ,0]), :]
    total = total.loc[total["quantile"] != "Q0", :]
    total = pd.DataFrame(total)
    total_damages = total.groupby('quantile').apply(lambda total: np.average(total.loc[~np.isnan(total.loc[: ,0]) ,0], weights=total.loc[~np.isnan(total.loc[: ,0]) ,3]))
    structure_damages = total.groupby('quantile').apply(lambda total: np.average(total.loc[~np.isnan(total.loc[: ,0]) ,1], weights=total.loc[~np.isnan(total.loc[: ,0]) ,3]))
    contents_damages = total.groupby('quantile').apply(lambda total: np.average(total.loc[~np.isnan(total.loc[: ,0]) ,2], weights=total.loc[~np.isnan(total.loc[: ,0]) ,3]))
    hh_per_quantile = total.groupby('quantile').sum()
    print(hh_per_quantile.loc[:,3 ])
    print(total_damages)
    
    order = ['Q1', 'Q2', 'Q3','Q4'] #,'Q5','Q6','Q7','Q8','Q9','Q10']
    colors = ['#FF9999', '#00BFFF','#C1FFC1','#CAE1FF','#FFDEAD']
    r = np.arange(4)
    barWidth = 0.25
    plt.figure(figsize=(10,7))
    plt.bar(r, np.array(pd.DataFrame(structure_damages).loc[order]).squeeze(), color=colors[0], edgecolor='white', width=barWidth, label="Structure")
    plt.bar(r, np.array(pd.DataFrame(contents_damages).loc[order]).squeeze(), bottom=np.array(pd.DataFrame(structure_damages).loc[order]).squeeze(), color=colors[1], edgecolor='white', width=barWidth, label='Contents')
    plt.legend(loc = 'upper left')
    plt.tick_params(labelbottom=True)
    plt.xticks(r, order)
    #plt.title(str(int(sum(total.loc[: ,3]))) + ' households \n ' + str(int(100 * sum(total.loc[: ,3])/sum(nb_hh))) + '% of households in ' + type_housing + ' housing')
    plt.title(str(int(sum(total.loc[: ,3]))) + ' households')
    #plt.savefig('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_per_hh_' + type_housing + '_' + option + '.png')  
    #plt.close()

def weighted_percentile(data, weights, perc):
    """
    perc : percentile in [0-1]!
    """
    ix = np.argsort(data)
    data = data[ix] # sort data
    weights = weights[ix] # sort weights
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights) # 'like' a CDF function
    return np.interp(perc, cdf, data)

"""
floods = ['FD_50yr']
path_data = "C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/"

option = "percent" #"absolu"

for item in floods:
    
    df = pd.DataFrame()
    
    type_flood = copy.deepcopy(item)
    data_flood = np.squeeze(pd.read_excel(path_data + item + ".xlsx"))
        
    formal_structure_damages = formal_structure_cost * depth_damage_function_structure(data_flood['flood_depth'])
    subsidized_structure_damages = param["subsidized_structure_value"] * depth_damage_function_structure(data_flood['flood_depth'])
    informal_structure_damages = param["informal_structure_value"] * depth_damage_function_structure(data_flood['flood_depth'])
    backyard_structure_damages = param["informal_structure_value"] * depth_damage_function_structure(data_flood['flood_depth'])
        
    formal_content_damages =  content_cost.formal * depth_damage_function_contents(data_flood['flood_depth'])
    subsidized_content_damages = content_cost.subsidized * depth_damage_function_contents(data_flood['flood_depth'])
    informal_content_damages = content_cost.informal * depth_damage_function_contents(data_flood['flood_depth'])
    backyard_content_damages = content_cost.backyard * depth_damage_function_contents(data_flood['flood_depth'])
        
    df['formal_structure_damages'] = formal_structure_damages
    df['subsidized_structure_damages'] = subsidized_structure_damages
    df['informal_structure_damages'] = informal_structure_damages
    df['backyard_structure_damages'] = backyard_structure_damages
    df['formal_content_damages'] = formal_content_damages
    df['informal_content_damages'] = informal_content_damages
    df['backyard_content_damages'] = backyard_content_damages
    df['subsidized_content_damages'] = subsidized_content_damages
    df["prop_flood_prone"] = data_flood["prop_flood_prone"]
    writer = pd.ExcelWriter('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + str(item) + '.xlsx')
    df.to_excel(excel_writer = writer)
    writer.save()
    
damages_50yr = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + 'FD_50yr' + '.xlsx')  
pop_damages_50yr = copy.deepcopy(damages_50yr.prop_flood_prone) * housing_types_grid.informal_grid_2011


"""

