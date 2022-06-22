import numpy as np
from cvxpy import *


def weekly_dispatcher(PV_power, Wind_power, load_profile, grid_price, SOC_ini):
    #print('Weekly Dispatch Started')


    #Wind cost is 0.0175 euro / kWh based on IRENA 2010 data
    #https://www.irena.org/-/media/Files/IRENA/Agency/Publication
    # /2012/RE_Technologies_Cost_Analysis-WIND_POWER.pdf
    Wind_opcost = 14.5 #euro/MWh

    Steam_opcost = 35 #euro/MWh
    length = np.size(PV_power)


    power_fromPV = Variable(length, nonneg=True)
    power_frombattery = Variable(length, nonneg = True)
    power_tobattery = Variable(length, nonneg = True)
    power_fromWind = Variable(length, nonneg=True)
    power_toGrid = Variable(length, nonneg = True)
    power_fromgrid = Variable(length, nonneg = True)
    SOC = Variable(length, nonneg= True)
    step_efficiency = Variable(length, nonneg = True)
    l1 = Variable(length, boolean= True)
    l2 = Variable(length, boolean = True)
    l3 = Variable(length, boolean=True)
    product = Variable(length, nonneg= True)



    SOC_max = 0.9
    SOC_min = 0.3
    #battery capacity is 30 MWh
    battery_capacity = 30
    #PV_capacity = 0.2  #MW

    constraints = []
    cost = 0


    #battery parameters

    for t in range(length):

        #cost +=  power_fromPV[t] + power_fromWind[t]  + power_fromgrid[t]

        #cost += grid_price[t]*power_fromgrid[t] + power_frombattery[t] * Steam_opcost + power_fromWind[t] * Wind_opcost

        cost += grid_price[t] * power_fromgrid[t] + power_frombattery[t] * Steam_opcost + power_fromWind[t] * Wind_opcost
        constraints += [power_fromPV[t] == PV_power[t]]
        constraints += [power_fromgrid[t] + power_fromPV[t] + power_frombattery[t] + power_fromWind[t] == load_profile[t] + power_toGrid[t] + power_tobattery[t]]
        constraints += [power_fromWind[t] == Wind_power[t]]
        constraints += [power_frombattery[t] <= 1.4]
        constraints += [power_tobattery[t] <= 5.4]

        constraints += [power_fromgrid[t] >= -30*(1-l2[t]), power_fromgrid[t] <= 30*l2[t]]
        constraints += [power_toGrid[t] <= 30*(1-l2[t]), power_toGrid[t] >= -30*(1-l2[t])]

        constraints += [power_frombattery[t] >= -10 * (1 - l3[t]), power_frombattery[t] <= 10 * l3[t]]
        constraints += [power_tobattery[t] <= 10 * (1 - l3[t]), power_tobattery[t] >= -10 * (1 - l3[t])]



        #Charging Constraints for Battery

        #constraints += [SOC[t] >= 0.6711 - 10 *(1-l1[t])]
        #constraints += [SOC[t] <= 0.6711 + 10*l1[t]]



       # constraints += [product[t] >= 0.55467*power_tobattery[t], product[t] <= 0.91689*power_tobattery[t]]
        #constraints += [product[t] >= 5.4*step_efficiency[t] +0.91689*power_tobattery[t] - 4.951206]
        #constraints += [product[t] <=  5.4*step_efficiency[t] +0.55467*power_tobattery[t] - 2.995218 ]
        constraints += [SOC[t] >= SOC_min, SOC[t] <= SOC_max]

        if (t==0):

            #if (SOC_ini <= 0.6711):
            #    constraints += [step_efficiency[t] == 0.8404]
            #else:
             #   constraints += [step_efficiency[t] == -1.731 *SOC_ini + 1.896]
            #constraints += [SOC[t] == SOC_ini + product[t] / battery_capacity - power_frombattery[t] / (0.2817*battery_capacity)]
            constraints += [
                SOC[t] == SOC_ini+ 0.6747 * power_tobattery[t] / battery_capacity - power_frombattery[t] / (
                            0.2817 * battery_capacity)]
            #constraints += [step_efficiency[t] == -0.6037 * SOC_ini + 1.098]
            #constraints += [product[t] <= (SOC_max - SOC_ini)*battery_capacity]
            #constraints += [power_frombattery[t]/0.2817 <= (SOC_ini - SOC_min)*battery_capacity]


        else:

            #SOC Update and Battery Efficiency Constraints
            #constraints += [SOC[t] == SOC[t - 1] + product[t] / battery_capacity - power_frombattery[t]/(0.2817*battery_capacity)]
            #constraints += [step_efficiency[t] == -0.6037 * SOC[t - 1] + 1.098]
            #constraints += [step_efficiency[t] <= -1.731 *SOC[t-1] + 1.896 + 10*(1-l1[t])]
            #constraints += [step_efficiency[t] >= -1.731 *SOC[t-1] + 1.896 - 10 * (1 - l1[t])]
            #constraints += [step_efficiency[t] <= 0.8404 + 10 * l1[t]]
            #constraints += [step_efficiency[t] >= 0.8404 - 10 * l1[t]]

            constraints += [SOC[t] == SOC[t - 1] + 0.6747*power_tobattery[t] / battery_capacity -  power_frombattery[t] / (0.2817*battery_capacity)]
            #constraints += [product[t] <= (SOC_max - SOC[t-1]) * battery_capacity]
           # constraints += [power_frombattery[t]/0.2817 <= (SOC[t-1] - SOC_min) * battery_capacity]
    #cost += PV_opcost*PV_capacity + CAPEX_battery*0.02

    simple_model = Problem(Minimize(cost), constraints)
    simple_model.solve(solver = GUROBI, verbose = False)
    return [power_fromPV.value,power_frombattery.value, power_fromWind.value, power_fromgrid.value, power_tobattery.value,
    power_toGrid.value, step_efficiency.value, SOC.value], simple_model.value