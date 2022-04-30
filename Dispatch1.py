import pandas as pd
import json
import numpy as np
from cvxpy import *
import matplotlib.pyplot as plt
import gurobipy
import matlab
import matlab.engine as me



#from cplex import *
#from cylp import *

### Reading PV power and converting it into W. Size is set to 2MW
df1 = pd.read_csv('PV_power.csv')
#print(df1.head(24))
PV_power = df1['System power generated | (kW)']/1000
#print(PV_power[0:48])

### Reading wind power. System size is set to 2MW
df2 = pd.read_csv('Wind_power.csv')
#print(df2.head(24))
Wind_power = df2['System power generated | (kW)']/1000
#print(Wind_power[0:48])

### Reading the load profile

#df3 = pd.read_json('v_opendata.json', orient = 'records')
#json_data = df3['data'][0]
#power_list = []

df4 = pd.read_excel('2021_ElectricityPrice.xlsx', sheet_name= 'Database', skiprows= range(3))
df4 = df4[['day', 'DE-LU']]
#print(df4.head())
#with open('temp.txt','w') as f1:
#    f1.write(str(json_data[0][0]))
#for i in range(len(json_data)):
#    total_list.append(json_data[i])
#for i in range(len(json_data)):
#    print(json_data[i]['internal_id_1'])

# Extracting only the sum values for load across all 3 phases
#power_list.extend(item for item in json_data if item['internal_id_2'] == 0)
#power_array = np.array([power_list[i]['values'] for i in range(len(power_list))])
#Averaging every 15 minutes
'''
power_15min = []
for i in range(len(power_array)):
   # print(i)
    power_15min.extend([np.mean(power_array[i].reshape(-1,15), axis = 1)])
#power_15min.extend(np.mean(np.mean(power_array[i].reshape(-1,15), axis = 1)) for i in range(len(power_array)))
print(np.shape(power_15min))
power_15min_step = np.repeat(power_15min,15, axis = 1)
print(np.shape(power_15min_step))
num_min = 24*60
power_1hr_step = np.reapeat
for i in range(np.shape(power_15min_step)[0]):
    plt.plot(power_15min_step[i][0:num_min])
plt.show()
'''
#load_1hr = []
#for i in range(len(power_array)):
   # print(i)
#    load_1hr.extend([np.mean(power_array[i].reshape(-1,60), axis = 1)])
#power_15min.extend(np.mean(np.mean(power_array[i].reshape(-1,15), axis = 1)) for i in range(len(power_array)))
#print(np.shape(power_15min))
#repeating the average for 60 minutes and converting to W
#load_1hr_step = np.repeat(load_1hr,60, axis = 1)

#converting kW to W
#load_1hr = np.divide(load_1hr,1000)
#np.savetxt('Load.csv', load_1hr, delimiter = ',')
#print(np.shape(load_1hr_step))
load_1hr = pd.read_csv('Load.csv')
print(load_1hr.head())
#num_min = 2*24*60
#for i in range(np.shape(load_1hr_step)[0]):
#    plt.plot(load_1hr_step[i][0:num_min])
#plt.show()
#print(check_list[0]['values'])
#print(id_list)
#print(len(check_list))
#print(json_data[:], file = f1)
#for i in range(len(df3['data'])):
#    print('ye check kro {}'.format(i))
#    print(type(df3['data']))
#    print(df3['data'][i], file = f1)
    #print(i, file = f1)
#print(df3['data'], file =f1)


#Wind cost is 0.0175 euro / kWh based on IRENA 2010 data
#https://www.irena.org/-/media/Files/IRENA/Agency/Publication
# /2012/RE_Technologies_Cost_Analysis-WIND_POWER.pdf
Wind_opcost = 0.0175  #euro/Wh
PV_opcost = 13 #euro/Wh
Steam_opcost = 35 #euro/Wh
grid_opcost = 100 #euro/Wh
export_opcost = 10 #euro/Wh

print('Hello ji 2',len(PV_power)/365)
length = int(2*len(PV_power)/365)
grid_price = df4[0:length]['DE-LU']
power_fromPV = Variable(length, nonneg=True)
power_frombattery = Variable(length, nonneg = True)
power_tobattery = Variable(length, nonneg = True)
power_fromWind = Variable(length, nonneg=True)
power_toGrid = Variable(length, nonneg = True)
power_fromgrid = Variable(length, nonneg = True)
import_switch = Variable(length, boolean= True)
export_switch = Variable(length, boolean = True)
SOC = Variable(length, nonneg = True)
mass_flow = Variable(length, nonneg = True)

SOC_ini = 0.9
SOC_max = 0.9
SOC_min = 0.3
#battery capacity is 30 MWh
battery_capacity = 30

#objective = Minimize(sum(PV_opcost*power_fromPV + power_frombattery*Steam_opcost + power_fromWind*Wind_opcost))
constraints = []
print('the size is {}'.format(len(PV_power)))


print("hello ji",len(load_1hr[0][0:length]))
initial_temp = []
initial_temp.extend([293,293])
cost = 0
eng = me.start_matlab()
s = eng.genpath("C:/Users/taimo/OneDrive - Universite de Lorraine\Bureau\DENSYS\Master Thesis\Thesis Work\Python Code")
eng.addpath(s, nargout=0)

#battery parameters
ca = 1100;
heater_eff = 0.9
T_in = 750+273
T_inf = 9.8+273
Pmax_fromheater = 5.4e6
m_flow_max = Pmax_fromheater/(ca*heater_eff*(T_in - T_inf))
#m_flow = power_fromheater/(ca*heater_eff*(T_in - T_inf));



for t in range(length):

    cost += grid_price[t]*(power_fromgrid[t] ) + PV_opcost * power_fromPV[t] + power_frombattery[t] * Steam_opcost + \
            power_fromWind[t] * Wind_opcost - 0.10*power_toGrid[t]*grid_price[t]
    constraints += [power_fromPV[t] == PV_power[t]]
    constraints += [power_fromgrid[t] + power_fromPV[t] + power_frombattery[t] + power_fromWind[t] == load_1hr[0][t]*1000 + power_toGrid[t] + power_tobattery[t]]
    constraints += [power_fromWind[t] == Wind_power[t]]
    constraints += [power_toGrid[t] <= 1]

    #Charging Constraints for Battery
    constraints += [mass_flow[t] == power_tobattery[t]*1e6 / (ca * heater_eff * (T_in - T_inf))]
    constraints += [mass_flow[t] <= m_flow_max]
    print('hello ji', matlab.double(power_tobattery[t]))
    SOC[t], mass_flow, temp_fromMAT = eng.Charger(matlab.double(power_tobattery[t].value), [293, 293])


    #constraints += [import_switch[t] + export_switch[t] == 1]

    if (t==0):
        constraints += [SOC[t] == SOC_ini + (power_tobattery[t] - power_frombattery[t])/battery_capacity]
        constraints += [power_tobattery[t] <= (SOC_max - SOC_ini)*battery_capacity]
        constraints += [power_frombattery[t] <= (SOC_ini - SOC_min)*battery_capacity]
    else:
        constraints += [SOC[t] == SOC[t-1] + (power_tobattery[t] - power_frombattery[t])/battery_capacity]
        constraints += [power_tobattery[t] <= (SOC_max - SOC[t-1]) * battery_capacity]
        constraints += [power_frombattery[t] <= (SOC[t-1] - SOC_min) * battery_capacity]

    #constraints += [SOC[t] <= 0.9 , SOC[t] >= 0.1]




#SOC = Parameter(length, nonneg=True)
#SOC = np.zeros(length + 1)
#for i in range(length):
#    if (i==0):
#        constraints += [SOC[i] == SOC_ini]
#    else:
#        constraints += [SOC[i] == (SOC[i-1] + power_frombattery/battery_capacity)]

#constraints = [constraint_Wind,constraint_Wind2, constraint_equality, constraint_PV, constraint_PV2]
simple_model = Problem(Minimize(cost), constraints)
simple_model.solve(solver = GUROBI, verbose = True)
print(installed_solvers())
solution_matrix = []
print('ye type he', type(power_fromPV))
plt.figure(1)
plt.plot(power_fromPV.value)
plt.plot(power_frombattery.value)
plt.plot(power_fromWind.value)
plt.plot(power_fromgrid.value)
plt.plot(-power_tobattery.value)
plt.plot(-power_toGrid.value)
plt.plot(load_1hr[0][0:length]*1000)
plt.legend(["PV", "From Battery", "Wind", "From Grid", "To Battery", "To Grid", "Load" ])

plt.figure(2)
plt.plot(SOC.value)


plt.figure(3)
plt.plot(mass_flow.value)
plt.show()
#plt.figure(2)
#plt.plot(SOC.value)
#plt.show()
#for i in range(len(df3['values'])):
#    print(i)
#df3 = df3[['data']]
#print(df3)
