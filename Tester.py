import cvxpy.atoms
import pandas as pd
import json
import numpy as np
from cvxpy import *
import matplotlib.pyplot as plt
import matlab.engine as me

### Reading PV power and converting it into W. Size is set to 2MW

df1 = pd.read_csv('PV_power.csv')
PV_power = df1['System power generated | (kW)']/1000
print('done 1')

Wind_opcost = 0.0175  #euro/Wh
PV_opcost = 13 #euro/Wh
Steam_opcost = 35 #euro/Wh
grid_opcost = 100 #euro/Wh
export_opcost = 10 #euro/Wh

print('Hello ji 2',len(PV_power)/365)
length = int(7*len(PV_power)/365)
#grid_price = df4[0:length]['DE-LU']
#print(len(grid_price))
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




p1 = Parameter()
p1.value = -1.828e-09
SOC_ini = 0.9
SOC_max = 0.9
SOC_min = 0.3
#battery capacity is 30 MWh
battery_capacity = 30

#objective = Minimize(sum(PV_opcost*power_fromPV + power_frombattery*Steam_opcost + power_fromWind*Wind_opcost))
constraints = []
#print('the size is {}'.format(len(load_1hr)))


#print("hello ji",(load_1hr[0][0:length]))
initial_temp = []
initial_temp.extend([293,293])
cost = 0
eng = me.start_matlab()
s = eng.genpath("C:/Users/taimo/OneDrive - Universite de Lorraine/Bureau/DENSYS/Master Thesis/Thesis Work/Python Code/PDE Solution")
eng.addpath(s, nargout=0)

#battery parameters
ca = 1100
heater_eff = 0.9
T_in = 750+273
T_inf = 9.8+273
Pmax_fromheater = 5.4e6
m_flow_max = Pmax_fromheater/(ca*heater_eff*(T_in - T_inf))
print(m_flow_max)
#m_flow = power_fromheater/(ca*heater_eff*(T_in - T_inf));
efficiency = pd.read_csv('efficiency.csv', header = None)
efficiency.columns = ['SOC', 'efficiency']
print(efficiency.head())
t = [1,2,3]
for t in t:
    print ('Iteration {}'.format(t))


    temp1 = step_efficiency[t].is_convex()
    temp2 = step_efficiency[t].is_concave()
    temp3 = step_efficiency[t].is_affine()
    #temp4 = power(SOC[t - 1], 0).is_convex()
    print('{}{}{}\n'.format(temp1, temp2, temp3))

    print('Convexity ')
    temp1 = power(SOC[t-1],3).is_convex()
    temp2 = (-1*power(SOC[t-1],2)).is_convex()
    temp3 = power(SOC[t - 1], 1).is_convex()
    temp4 = power(SOC[t - 1], 0).is_convex()
    print('{}{}{}{} '.format(temp1, temp2, temp3, temp4))

    print('Concavity ')
    temp1 = power(SOC[t-1],3).is_concave()
    temp2 = (-1*power(SOC[t-1],2)).is_concave()
    temp3 = power(SOC[t - 1], 1).is_concave()
    temp4 = power(SOC[t - 1], 0).is_concave()
    print('{}{}{}{}'.format(temp1, temp2, temp3, temp4))

    print('Affinity ')
    temp1 = power(SOC[t - 1], 3).is_affine()
    temp2 = power(SOC[t - 1], 2).is_affine()
    temp3 = power(SOC[t - 1], 1).is_affine()
    temp4 = power(SOC[t - 1], 0).is_affine()
    print('{}{}{}{}\n'.format(temp1, temp2, temp3, temp4))

    expr = square((step_efficiency[t] + power_tobattery[t]))
    print('final ye he',expr.is_convex())
    print('hello ji', (-square(step_efficiency[t])).is_convex())
    #expression = -1.828e-9*power(SOC[t-1],3) + 2.741e-7 * square(SOC[t-1]) + 1.076e-6 * SOC[t-1] + 0.685
    #expression = (step_efficiency[t] + 0.4213 * power(SOC[t-1],3) <=  0.101 * square(SOC[t - 1]) + 0.0009924 * SOC[t - 1] + 0.6085)
    expression = (step_efficiency[t]  <= (-0.5163 * square(SOC[t - 1]) + 0.242 * SOC[t - 1] + 0.5889) )

    print('\n Expression Validity {}'.format(expression.is_dcp()))
   # print('\n Expression Concavity {}'.format(expression.is_concave()))
    #print('\n Expression Affinity {}'.format(expression.is_affine()))


#for t in range(length):
    #g = (step_efficiency[t] >= power(SOC[t-1], 3)).is_dcp()
    #print(g)
#    print((-1.234*power(SOC[t-1],3)).is_concave())
#    print((-1.234 * power(SOC[t-1],3) >= step_efficiency[t]).is_dcp())
#    print(power[SOC])
'''

simple_model.solve(solver = GUROBI, verbose = True)
print(installed_solvers())
solution_matrix = []
print('ye type he', type(power_fromPV))
plt.figure(1)
x = np.linspace(0,length-1,length)
print(x)
hello_ji = power_fromPV.value
print(type(hello_ji))
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


plt.figure(3)
plt.step(x,mass_flow.value)
plt.show()
#plt.figure(2)
#plt.plot(SOC.value)
#plt.show()
#for i in range(len(df3['values'])):
#    print(i)
#df3 = df3[['data']]
#print(df3)
'''