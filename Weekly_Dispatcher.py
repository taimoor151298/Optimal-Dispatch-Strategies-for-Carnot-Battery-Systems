import numpy as np
from cvxpy import *


def weekly_dispatcher(PV_power, Wind_power, load_profile, grid_price, SOC_ini):
    #print('Weekly Dispatch Started')


    #Wind cost is 0.0175 euro / kWh based on IRENA 2010 data
    #https://www.irena.org/-/media/Files/IRENA/Agency/Publication
    # /2012/RE_Technologies_Cost_Analysis-WIND_POWER.pdf
    Wind_opcost = 0.0175  #euro/Wh
    PV_opcost = 13 #euro/Wh
    Steam_opcost = 35 #euro/Wh
    length = np.size(PV_power)


    power_fromPV = Variable(length, nonneg=True)
    power_frombattery = Variable(length, nonneg = True)
    power_tobattery = Variable(length, nonneg = True)
    power_fromWind = Variable(length, nonneg=True)
    power_toGrid = Variable(length, nonneg = True)
    power_fromgrid = Variable(length, nonneg = True)
    SOC = Variable(length, nonneg= True)
    mass_flow = Variable(length, nonneg = True)
    step_efficiency = Variable(length, nonneg = True)
    l1 = Variable(length, boolean= True)
    product = Variable(length, nonneg= True)



    SOC_max = 0.9
    SOC_min = 0.3
    #battery capacity is 30 MWh
    battery_capacity = 30


    constraints = []
    cost = 0


    #battery parameters

    for t in range(length):

        cost += grid_price[t]*power_fromgrid[t] + PV_opcost * power_fromPV[t] + power_frombattery[t] * Steam_opcost + \
                power_fromWind[t] * Wind_opcost - power_toGrid[t]*grid_price[t]
        constraints += [power_fromPV[t] == PV_power[t]]
        constraints += [power_fromgrid[t] + power_fromPV[t] + power_frombattery[t] + power_fromWind[t] == load_profile[t] + power_toGrid[t] + power_tobattery[t]]
        constraints += [power_fromWind[t] == Wind_power[t]]
        constraints += [power_frombattery[t] <= 1.4]


        #Charging Constraints for Battery

        constraints += [SOC[t] >= 0.4988 - 10 *(1-l1[t])]
        constraints += [SOC[t] <= 0.4988 + 10*l1[t]]


        #constraints += [step_efficiency[t] == 0.6037]
        constraints += [product[t] >= 0.47853*power_tobattery[t], product[t] <= 0.6238*power_tobattery[t]]
        constraints += [product[t] >= 5.4*step_efficiency[t] +0.6238*power_tobattery[t] - 3.36852 ]
        constraints += [product[t] <=  5.4*step_efficiency[t] +0.47853*power_tobattery[t] - 2.584062 ]
        #constraints += [SOC[t] >= SOC_min, SOC[t] <= SOC_max]

        if (t==0):

            #constraints += [SOC[t] == SOC_ini + power_tobattery[t]/battery_capacity - power_frombattery[t]/battery_capacity]
            if (SOC_ini <= 0.4988):
                constraints += [step_efficiency[t] == 0.6037]
            else:
                constraints += [step_efficiency[t] == -0.3623 * SOC_ini + 0.8046]

            #constraints += [product[t] == -1.97046 * SOC_ini + 0.65313 * power_tobattery[t] + 0.591138]
            constraints += [SOC[t] == SOC_ini + product[t] / battery_capacity - power_frombattery[t] / (0.4*battery_capacity)]
            constraints += [product[t] <= (SOC_max - SOC_ini)*battery_capacity]
            constraints += [power_frombattery[t]/0.4 <= (SOC_ini - SOC_min)*battery_capacity]


        else:


            constraints += [SOC[t] == SOC[t - 1] + product[t] / battery_capacity - power_frombattery[t] / (0.4*battery_capacity)]
            constraints += [step_efficiency[t] <= -0.3623*SOC[t-1] + 0.8046 + 10*(1-l1[t])]
            constraints += [step_efficiency[t] >= -0.3623 * SOC[t - 1] + 0.8046 - 10 * (1 - l1[t])]
            constraints += [step_efficiency[t] <= 0.6037 + 10 * l1[t]]
            constraints += [step_efficiency[t] >= 0.6037 - 10 * l1[t]]

           # constraints += [SOC[t] == SOC[t - 1] + power_tobattery[t] / battery_capacity -  power_frombattery[t] / battery_capacity]
            constraints += [product[t] <= (SOC_max - SOC[t-1]) * battery_capacity]
            constraints += [power_frombattery[t]/0.4<= (SOC[t-1] - SOC_min) * battery_capacity]


    simple_model = Problem(Minimize(cost), constraints)
    simple_model.solve(solver = GUROBI, verbose = False, reoptimize = True)
    return [power_fromPV.value,power_frombattery.value, power_fromWind.value, power_fromgrid.value, power_tobattery.value,
    power_toGrid.value, step_efficiency.value, SOC.value], simple_model.value
