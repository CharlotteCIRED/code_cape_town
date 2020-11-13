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
        
    formal_structure_damages = data_flood["prop_flood_prone"] * formal_structure_cost * depth_damage_function_structure(data_flood['flood_depth'])
    subsidized_structure_damages = data_flood["prop_flood_prone"] * param["subsidized_structure_value"] * depth_damage_function_structure(data_flood['flood_depth'])
    informal_structure_damages = data_flood["prop_flood_prone"] * param["informal_structure_value"] * depth_damage_function_structure(data_flood['flood_depth'])
    backyard_structure_damages = data_flood["prop_flood_prone"] * param["informal_structure_value"] * depth_damage_function_structure(data_flood['flood_depth'])
        
    formal_content_damages =  data_flood["prop_flood_prone"] * content_cost.formal * depth_damage_function_contents(data_flood['flood_depth'])
    subsidized_content_damages = data_flood["prop_flood_prone"] * content_cost.subsidized * depth_damage_function_contents(data_flood['flood_depth'])
    informal_content_damages = data_flood["prop_flood_prone"] * content_cost.informal * depth_damage_function_contents(data_flood['flood_depth'])
    backyard_content_damages = data_flood["prop_flood_prone"] * content_cost.backyard * depth_damage_function_contents(data_flood['flood_depth'])
        
    df['formal_structure_damages'] = formal_structure_damages
    df['subsidized_structure_damages'] = subsidized_structure_damages
    df['informal_structure_damages'] = informal_structure_damages
    df['backyard_structure_damages'] = backyard_structure_damages
    df['formal_content_damages'] = formal_content_damages
    df['informal_content_damages'] = informal_content_damages
    df['backyard_content_damages'] = backyard_content_damages
    df['subsidized_content_damages'] = subsidized_content_damages
    writer = pd.ExcelWriter('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_' + str(item) + '.xlsx')
    df.to_excel(excel_writer = writer)
    writer.save()
   
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

annualized_damages_per_hh = annualize_damages([damages_5yr, damages_10yr, damages_20yr, damages_50yr, damages_75yr, damages_100yr, damages_200yr, damages_250yr, damages_500yr, damages_1000yr])

if option == "percent":
    income_class = pd.DataFrame()
    income_class["formal"] = np.argmax(initial_state_households[0,:,:], 0)
    income_class["backyard"] = np.argmax(initial_state_households[1,:,:], 0)
    income_class["informal"] = np.argmax(initial_state_households[2,:,:], 0)
    income_class["subsidized"] = np.argmax(initial_state_households[3,:,:], 0)
    annualized_damages_per_hh.columns
    annualized_damages_per_hh.subsidized_structure_damages = annualized_damages_per_hh.subsidized_structure_damages / income_net_of_commuting_costs[0, :]
    annualized_damages_per_hh.subsidized_content_damages = annualized_damages_per_hh.subsidized_content_damages / income_net_of_commuting_costs[0, :]

    for i in range(0, len(annualized_damages_per_hh.formal_structure_damages)):
        print(i)
        annualized_damages_per_hh.formal_structure_damages[i] = annualized_damages_per_hh.formal_structure_damages[i] / income_net_of_commuting_costs[np.array(income_class["formal"])[i], i]
        annualized_damages_per_hh.formal_content_damages[i] = annualized_damages_per_hh.formal_content_damages[i] / income_net_of_commuting_costs[int(income_class["formal"][i]), i]
        
        annualized_damages_per_hh.backyard_structure_damages[i] = annualized_damages_per_hh.backyard_structure_damages[i] / income_net_of_commuting_costs[np.array(income_class["backyard"])[i], i]
        annualized_damages_per_hh.backyard_content_damages[i] = annualized_damages_per_hh.backyard_content_damages[i] / income_net_of_commuting_costs[int(income_class["backyard"][i]), i]
        
        annualized_damages_per_hh.informal_structure_damages[i] = annualized_damages_per_hh.informal_structure_damages[i] / income_net_of_commuting_costs[np.array(income_class["informal"])[i], i]
        annualized_damages_per_hh.informal_content_damages[i] = annualized_damages_per_hh.informal_content_damages[i] / income_net_of_commuting_costs[int(income_class["informal"][i]), i]


    annualized_damages_per_hh["nb_hh_formal"] = initial_state_households_housing_types[0, :]
    annualized_damages_per_hh["nb_hh_subsidized"] = initial_state_households_housing_types[3, :]
    annualized_damages_per_hh["nb_hh_informal"] = initial_state_households_housing_types[2, :]
    annualized_damages_per_hh["nb_hh_backyard"] = initial_state_households_housing_types[1, :]

df_subsidized = annualized_damages_per_hh.loc[annualized_damages_per_hh["nb_hh_subsidized"] > 0, :]
df_subsidized["total_damages"] = df_subsidized.subsidized_structure_damages + df_subsidized.subsidized_content_damages
if option == "percent":
    df_subsidized = df_subsidized.loc[df_subsidized["total_damages"] > 0.03, :]
elif option == "absolu":
    df_subsidized = df_subsidized.loc[df_subsidized["total_damages"] > 20, :]

df_formal = annualized_damages_per_hh.loc[annualized_damages_per_hh["nb_hh_formal"] > 0, :]
df_formal["total_damages"] = df_formal.formal_structure_damages + df_formal.formal_content_damages
df_formal = df_formal.loc[~np.isnan(df_formal["total_damages"]), :]
if option == "percent":
    df_formal = df_formal.loc[df_formal["total_damages"] > 0.03, :]
elif option == "absolu":
    df_formal = df_formal.loc[df_formal["total_damages"] > 20, :]

df_informal = annualized_damages_per_hh.loc[annualized_damages_per_hh["nb_hh_informal"] > 0, :]
df_informal["total_damages"] = df_informal.informal_structure_damages + df_informal.informal_content_damages
df_informal = df_informal.loc[~np.isnan(df_informal["total_damages"]), :]
df_informal = df_informal.loc[df_informal["total_damages"] > 0, :]
if option == "percent":
    df_informal = df_informal.loc[df_informal["total_damages"] > 0.03, :]
elif option == "absolu":
    df_informal = df_informal.loc[df_informal["total_damages"] > 20, :]

df_backyard = annualized_damages_per_hh.loc[annualized_damages_per_hh["nb_hh_backyard"] > 0, :]
df_backyard["total_damages"] = df_backyard.backyard_structure_damages + df_backyard.backyard_content_damages
df_backyard = df_backyard.loc[~np.isnan(df_backyard["total_damages"]), :]
df_backyard = df_backyard.loc[df_backyard["total_damages"] > 0, :]
if option == "percent":
    df_backyard = df_backyard.loc[df_backyard["total_damages"] > 0.03, :]
elif option == "absolu":
    df_backyard = df_backyard.loc[df_backyard["total_damages"] > 20, :]

#1. Histogram

#df_subsidized.total_damages.plot(kind='hist', bins = 200, weights=df_subsidized.nb_hh_subsidized)
#plt.tick_params(labelbottom=True)

#df_formal.total_damages.plot(kind='hist', bins = 200, weights=df_formal.nb_hh_formal)
#plt.tick_params(labelbottom=True)

#df_informal.total_damages.plot(kind='hist', bins = 200, weights=df_informal.nb_hh_informal)
#plt.tick_params(labelbottom=True)

#df_backyard.total_damages.plot(kind='hist', bins = 200, weights=df_backyard.nb_hh_backyard)
#plt.tick_params(labelbottom=True)

#2. Bar plot

#Subsidized

plot_damages_decile(df_subsidized["total_damages"], df_subsidized["nb_hh_subsidized"], df_subsidized.subsidized_structure_damages, df_subsidized.subsidized_content_damages, 'subsidized', name, initial_state_households_housing_types[3, :])
plot_damages_decile(df_formal["total_damages"], df_formal["nb_hh_formal"], df_formal.formal_structure_damages, df_formal.formal_content_damages, 'formal', name, initial_state_households_housing_types[0, :])
plot_damages_decile(df_informal["total_damages"], df_informal["nb_hh_informal"], df_informal.informal_structure_damages, df_informal.informal_content_damages, 'informal', name, initial_state_households_housing_types[2, :])
plot_damages_decile(df_backyard["total_damages"], df_backyard["nb_hh_backyard"], df_backyard.backyard_structure_damages, df_backyard.backyard_content_damages, 'backyard', name, initial_state_households_housing_types[1, :])

def plot_damages_decile(total_damages, nb_hh, structure_damages, content_damages, housing_type, name, households_housing_types):

    array_percentile = weighted_percentile(np.array(total_damages), np.array(nb_hh), np.arange(0, 1, 0.1))
    
    df = pd.DataFrame(data = {'total_damages' : total_damages, 'nb_hh':nb_hh, 'structure_damages':structure_damages, 'content_damages':content_damages, 'quantile': np.empty(len(total_damages))})
    for i in range(1, len(array_percentile)):
        df["quantile"][(total_damages < array_percentile[i]) & (total_damages >= array_percentile[i - 1])] = "Q" + str(i)
    df["quantile"][(total_damages > array_percentile[9])] = "Q10"

    total_damages = df.groupby('quantile').apply(lambda df: np.average(df.total_damages, weights=df.nb_hh))
    structure_damages = df.groupby('quantile').apply(lambda df: np.average(df.structure_damages, weights=df.nb_hh))
    contents_damages = df.groupby('quantile').apply(lambda df: np.average(df.content_damages, weights=df.nb_hh))

    order = ['Q1', 'Q2', 'Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10']
    colors = ['#FF9999', '#00BFFF','#C1FFC1','#CAE1FF','#FFDEAD']
    r = np.arange(10)
    barWidth = 0.25
    plt.figure(figsize=(10,7))
    plt.bar(r, np.array(pd.DataFrame(structure_damages).loc[order]).squeeze(), color=colors[0], edgecolor='white', width=barWidth, label="Structure")
    plt.bar(r, np.array(pd.DataFrame(contents_damages).loc[order]).squeeze(), bottom=np.array(pd.DataFrame(structure_damages).loc[order]).squeeze(), color=colors[1], edgecolor='white', width=barWidth, label='Contents')
    plt.legend(loc = 'upper left')
    plt.tick_params(labelbottom=True)
    plt.xticks(r, order)
    plt.title(str(int(sum(nb_hh))) + ' households \n ' + str(int(100 * sum(nb_hh)/sum(households_housing_types))) + '% of households in ' + housing_type +' housing')
    plt.savefig('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/damages_per_hh_' + housing_type + '_v2.png')  
    plt.close()



def weighted_percentile(data, weights, perc):
    """
    perc : percentile in [0-1]!
    """
    ix = np.argsort(data)
    data = data[ix] # sort data
    weights = weights[ix] # sort weights
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights) # 'like' a CDF function
    return np.interp(perc, cdf, data)

