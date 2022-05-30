import cvxpy.atoms
import pandas as pd
import json
import numpy as np
from cvxpy import *
import matplotlib.pyplot as plt


### Reading PV power and converting it into W. Size is set to 2MW

df1 = pd.read_csv('PV_power.csv')
PV_power = df1['System power generated | (kW)']/1000
print('done 1')

### Reading wind power. System size is set to 2MW
df2 = pd.read_csv('Wind_power.csv')
Wind_power = df2['System power generated | (kW)']/1000
print('done 2')

### Reading the load profile


df4 = pd.read_excel('2021_ElectricityPrice.xlsx', sheet_name= 'Database', skiprows= range(3))
df4 = df4[['day', 'DE-LU']]
print('done 3')

load_1hr = pd.read_csv('Load.csv', header = None)
print(load_1hr.head())
print('done 4')
#Wind cost is 0.0175 euro / kWh based on IRENA 2010 data
#https://www.irena.org/-/media/Files/IRENA/Agency/Publication
# /2012/RE_Technologies_Cost_Analysis-WIND_POWER.pdf
Wind_opcost = 0.0175  #euro/Wh
PV_opcost = 13 #euro/Wh
Steam_opcost = 35 #euro/Wh
grid_opcost = 100 #euro/Wh
export_opcost = 10 #euro/Wh

print('Hello ji 2',len(PV_power)/365)
length = int(364*len(PV_power)/365)
grid_price = df4[0:length]['DE-LU']
print(len(grid_price))
power_fromPV = Variable(length, nonneg=True)
power_frombattery = Variable(length, nonneg = True)
power_tobattery = Variable(length, nonneg = True)
power_fromWind = Variable(length, nonneg=True)
power_toGrid = Variable(length, nonneg = True)
power_fromgrid = Variable(length, nonneg = True)
import_switch = Variable(length, boolean= True)
export_switch = Variable(length, boolean = True)
SOC = Variable(length, nonneg= True)
mass_flow = Variable(length, nonneg = True)
step_efficiency = Variable(length, nonneg = True)
l1 = Variable(length, boolean= True)
#l2 = Variable(length, boolean = True)
#l3 = Variable(length, boolean = True)
product = Variable(length, nonneg= True)


SOC_ini = 0.9
SOC_max = 0.9
SOC_min = 0.3
#battery capacity is 30 MWh
battery_capacity = 30

#objective = Minimize(sum(PV_opcost*power_fromPV + power_frombattery*Steam_opcost + power_fromWind*Wind_opcost))
constraints = []
print('the size is {}'.format(len(load_1hr)))


print("hello ji",(load_1hr[0][0:length]))
#initial_temp = []
#initial_temp.extend([293,293])
cost = 0
#eng = me.start_matlab()
#s = eng.genpath("C:/Users/taimo/OneDrive - Universite de Lorraine/Bureau/DENSYS/Master Thesis/Thesis Work/Python Code/PDE Solution")
#eng.addpath(s, nargout=0)

#battery parameters
ca = 1100
heater_eff = 0.95
T_in = 750+273
T_inf = 9.8+273
Pmax_toheater = 5.4e6
m_flow_max = Pmax_toheater/(ca*heater_eff*(T_in - T_inf))
print(m_flow_max)
#m_flow = power_fromheater/(ca*heater_eff*(T_in - T_inf));
#efficiency = pd.read_csv('efficiency.csv', header = None)
#efficiency.columns = ['SOC', 'efficiency']
#print(efficiency.head())

for t in range(length):
    #g = (step_efficiency[t] >= power(SOC[t-1], 3)).is_dcp()
    #print(g)
    #print((-1.234*power(SOC[t-1],3)).is_concave())
    #print((-1.234 * power(SOC[t-1],3) >= step_efficiency[t]).is_dcp())
    cost += grid_price[t]*(power_fromgrid[t]) + PV_opcost * power_fromPV[t] + power_frombattery[t] * Steam_opcost + \
            power_fromWind[t] * Wind_opcost - power_toGrid[t]*grid_price[t]
    #cost += step_efficiency[t]
    constraints += [power_fromPV[t] == PV_power[t]]
    constraints += [power_fromgrid[t] + power_fromPV[t] + power_frombattery[t] + power_fromWind[t] == load_1hr[0][t]*1000 + power_toGrid[t] + power_tobattery[t]]
    constraints += [power_fromWind[t] == Wind_power[t]]
    constraints += [power_frombattery[t] <= 1.4]


    #Charging Constraints for Battery

    constraints += [SOC[t] >= 0.4988 - 10 * (1 - l1[t])]
    constraints += [SOC[t] <= 0.4988 + 10 * l1[t]]

    constraints += [product[t] >= 0.47853 * power_tobattery[t], product[t] <= 0.6238 * power_tobattery[t]]
    constraints += [product[t] >= 5.4 * step_efficiency[t] + 0.6238 * power_tobattery[t] - 3.36852]
    constraints += [product[t] <= 5.4 * step_efficiency[t] + 0.47853 * power_tobattery[t] - 2.584062]
    constraints += [SOC[t] <= SOC_max, SOC[t]>= SOC_min]
    if (t==0):

        #constraints += [SOC[t] == SOC_ini + power_tobattery[t]/battery_capacity - power_frombattery[t]/battery_capacity]
        if (SOC_ini <= 4988):
            constraints += [step_efficiency[t] == 0.6037]
        else:
         constraints += [step_efficiency[t] == -0.3623*SOC_ini + 0.8046]
        #constraints += [product[t] == -1.97046 * SOC_ini + 0.65313 * power_tobattery[t] + 0.591138]
        constraints += [SOC[t] == SOC_ini + product[t] / battery_capacity - power_frombattery[t] / (0.4*battery_capacity)]
        #constraints += [product[t] <= (SOC_max - SOC_ini)*battery_capacity]
        #constraints += [power_frombattery[t]/0.4 <= (SOC_ini - SOC_min)*battery_capacity]


    else:


        constraints += [SOC[t] == SOC[t - 1] + product[t] / battery_capacity - power_frombattery[t]/(0.4 * battery_capacity)]
        constraints += [step_efficiency[t] <= -0.3623*SOC[t-1] + 0.8046 + 10*(1-l1[t])]
        constraints += [step_efficiency[t] >= -0.3623 * SOC[t - 1] + 0.8046 - 10 * (1 - l1[t])]
        constraints += [step_efficiency[t] <= 0.6037 + 10 * l1[t]]
        constraints += [step_efficiency[t] >= 0.6037 - 10 * l1[t]]

       # constraints += [SOC[t] == SOC[t - 1] + power_tobattery[t] / battery_capacity -  power_frombattery[t] / battery_capacity]
        #constraints += [product[t] <= (SOC_max - SOC[t-1]) * battery_capacity]
        #constraints += [power_frombattery[t]/0.4 <= (SOC[t-1] - SOC_min) * battery_capacity]


#constraints = [constraint_Wind,constraint_Wind2, constraint_equality, constraint_PV, constraint_PV2]
simple_model = Problem(Minimize(cost), constraints)


simple_model.solve(solver = GUROBI, verbose = True)
print(installed_solvers())
solution_matrix = []
print('ye type he', type(power_fromPV))
plt.figure(1)
x = np.linspace(0,length-1,length)
#print(x)
hello_ji = power_fromPV.value
#print(type(hello_ji))
plt.step(x,power_fromPV.value)
plt.step(x,power_frombattery.value)
plt.step(x,power_fromWind.value)
plt.step(x,power_fromgrid.value)
plt.step(x,-power_tobattery.value)
plt.step(x,-power_toGrid.value)
plt.step(x,load_1hr[0][0:length]*1000)
plt.legend(["PV", "From Battery", "Wind", "From Grid", "To Battery", "To Grid", "Load" ])

plt.figure(2)
plt.step(x,SOC.value)


#plt.figure(3)
#plt.step(x,mass_flow.value)


plt.figure(4)
plt.step(x,step_efficiency.value)
#plt.step(x, product.value/power_tobattery.value)

plt.figure(5)
plt.subplot(1,3,1)
plt.step(x, SOC.value, 'r')
plt.ylim([0, 1])

plt.title('SOC with approximated product')

plt.subplot(1,3,2)

SOC_original = np.zeros(length)
for t in range(length):
    if(t==0):
        print('in loop', step_efficiency[t].value)
        SOC_original[t] = SOC_ini + step_efficiency[t].value*power_tobattery[t].value/battery_capacity - power_frombattery[t].value  \
                          /(0.4*battery_capacity)
    else:
        #print('again in loop',step_efficiency[t].value*power_tobattery[t].value )
        SOC_original[t] = SOC_original[t-1] + step_efficiency[t].value*power_tobattery[t].value /battery_capacity - power_frombattery[t].value\
                          /(0.4*battery_capacity)

plt.step(x, SOC_original, 'g')
plt.ylim([0, 1])
plt.title('SOC with actual product')

plt.subplot(1,3,3)

SOC_approx = np.around(SOC.value, decimals = 2)
SOC_original = np.around(SOC_original, decimals =2 )
#print('actual product', original)
#print('approx product', product.value)
#temp_array = np.around(np.array(product.value), decimals = 2)


error = np.abs(np.divide(SOC_original - SOC_approx,SOC_original ,
              out=np.zeros_like(SOC_original - SOC_approx),  where = SOC_original!=0))*100
plt.step(x, error, 'g')
plt.title('Error')

plt.show()
#plt.figure(2)
#plt.plot(SOC.value)
#plt.show()
#for i in range(len(df3['values'])):
#    print(i)
#df3 = df3[['data']]
#print(df3)
