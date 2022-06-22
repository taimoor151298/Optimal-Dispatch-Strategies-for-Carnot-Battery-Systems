from Weekly_Dispatcher import weekly_dispatcher
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


### Reading PV power and converting it into W. Size is set to 2MW

PV_capacity = 4  #MW
PV_opcost = 17 * 1e3 * 0.93  # euro/MW-year
df1 = pd.read_csv('PV_power.csv')
PV_power = PV_capacity*np.array(df1['System power generated | (kW)']/1000)/0.2

print('PV Power Profile Loaded')

### Reading wind power. System size is set to 2MW
df2 = pd.read_csv('Wind_power.csv')
Wind_power = 2*np.array(df2['System power generated | (kW)']/1000)
print('Wind Power Profile Loaded')

### Reading the load profile


df4 = pd.read_excel('2021_ElectricityPrice.xlsx', sheet_name= 'Database', skiprows= range(3))
df4 = df4[['day', 'DE-LU']]
grid_price = np.array(df4['DE-LU'])
print('Electricity Prices Loaded')

load_profile = np.array(1000*(pd.read_csv('Load.csv', header = None)[0]))
print('Load Profile Loaded')

n_days = 7
n_steps = 52

hours = int(n_steps*n_days*24)

total_cost = 0


power_fromPV = []
power_frombattery = []
power_fromWind = []
power_fromgrid= []
power_tobattery = []
power_toGrid = []
step_efficiency = []
SOC = []
SOC_ini = 0.9
SOC_store = [SOC_ini]


for i in range(n_steps):
    starting_hour = i*n_days*24
    ending_hour = (i+1)*n_days*24

    solution_matrix, step_cost = weekly_dispatcher(PV_power[starting_hour:ending_hour], Wind_power[starting_hour:ending_hour],
                      load_profile[starting_hour:ending_hour],grid_price[starting_hour:ending_hour], SOC_ini)
    total_cost += step_cost
    power_fromPV = np.append(power_fromPV, solution_matrix[0])
    power_frombattery = np.append(power_frombattery, solution_matrix[1])
    power_fromWind = np.append(power_fromWind, solution_matrix[2])
    power_fromgrid = np.append(power_fromgrid, solution_matrix[3])
    power_tobattery = np.append(power_tobattery, solution_matrix[4])
    power_toGrid = np.append(power_toGrid, solution_matrix[5])
    step_efficiency = np.append(step_efficiency, solution_matrix[6])
    SOC = np.append(SOC, solution_matrix[7])
    SOC_ini = SOC[-1]
    SOC_store = np.append(SOC_store,SOC_ini)
    print('Week {} : {}'.format( i+1, step_cost))


plt.figure(1)


starting_hour = 0
ending_hour = n_days*n_steps*24
x = np.linspace(starting_hour,ending_hour-1,ending_hour)
plt.legend(["PV", "From Battery", "Wind", "From Grid", "To Battery", "To Grid", "Load" ])

plt.step(x,power_fromPV[starting_hour:ending_hour])
plt.step(x,power_frombattery[starting_hour:ending_hour])
plt.step(x,power_fromWind[starting_hour:ending_hour])
plt.step(x,power_fromgrid[starting_hour:ending_hour])
plt.step(x,-power_tobattery[starting_hour:ending_hour])
plt.step(x,-power_toGrid[starting_hour:ending_hour])
plt.step(x,load_profile[starting_hour:ending_hour])
plt.legend(["PV", "From Battery", "Wind", "From Grid", "To Battery", "To Grid", "Load" ])
plt.xlabel('Time (hours)')
plt.ylabel('Power (MW)')
plt.figure(2)
plt.step(x,SOC[starting_hour:ending_hour])


#plt.figure(3)
#plt.step(x,mass_flow.value)


#plt.figure(4)
#plt.step(x,step_efficiency[starting_hour:ending_hour])
#plt.step(x, product.value/power_tobattery.value)
'''

plt.figure(5)
plt.subplot(1,3,1)
plt.step(x, SOC[starting_hour:ending_hour], 'r')
plt.ylim([0, 1])

plt.title('SOC with approximated product')

plt.subplot(1,3,2)

battery_capacity = 30
SOC_original_total = []
SOC_original_step = np.zeros(hours//n_steps)
step_efficiency_step = []
power_tobattery_step = []
power_frombattery_step = []
print(SOC_store)
for i in range(n_steps):
    step_efficiency_step = step_efficiency[n_days*24*i:(i + 1) * n_days*24]
    power_tobattery_step = power_tobattery[i * n_days*24:(i + 1) * n_days*24]
    power_frombattery_step = power_frombattery[i * n_days*24:(i + 1) * n_days*24]

    for t in range(hours//n_steps):
        if (t == 0):
            SOC_original_step[t] = SOC_store[i] + step_efficiency_step[t]*power_tobattery_step[t]/battery_capacity - power_frombattery_step[t]  \
                              /(0.2817*battery_capacity)
        else:
            #print('again in loop',step_efficiency[t].value*power_tobattery[t].value )
            SOC_original_step[t] = SOC_original_step[t-1] + step_efficiency_step[t]*power_tobattery_step[t]/battery_capacity - power_frombattery_step[t]\
                              /(0.2817*battery_capacity)
    SOC_original_total = np.append(SOC_original_total, SOC_original_step)
plt.step(x, SOC_original_total[starting_hour:ending_hour], 'g')
plt.ylim([0, 1])
plt.title('SOC with actual product')

plt.subplot(1,3,3)

SOC_approx = np.around(SOC, decimals = 3)
SOC_original = np.around(SOC_original_total, decimals =3)

error = np.abs(np.divide(SOC_original - SOC_approx,SOC_original ,
              out=np.zeros_like(SOC_original - SOC_approx),  where = SOC_original!=0))*100
plt.step(x, error[starting_hour:ending_hour], 'g')
plt.title('Error')
'''

CAPEX_battery = 123222.71

total_cost += PV_opcost*PV_capacity + CAPEX_battery*0.02
print('Optimal Value of Objective Function', total_cost)
#np.set_printoptions(precision=3)
#print(np.around(SOC,4))

print(np.shape(power_fromgrid))
print(np.shape(grid_price[starting_hour:ending_hour]))
final_df = pd.DataFrame(data = [power_fromPV, power_fromWind, power_fromgrid, power_toGrid, power_frombattery, power_tobattery, grid_price[starting_hour:ending_hour],
                                load_profile[starting_hour:ending_hour], step_efficiency, SOC[starting_hour:ending_hour]])
final_df = final_df.T
final_df.index = np.arange(1, len(final_df)+1)
final_df.loc['Column_Total']= final_df.sum(numeric_only=True, axis=0)
final_df.columns = ['PV', 'Wind', 'From Grid', 'To Grid', 'From Battery', 'To Battery', 'Grid Price', 'Load Profile', 'Efficiency', 'SOC']
final_df.to_csv('MinGrid_8MW.csv')



print(sum(power_frombattery))
print(sum(power_fromWind))
print(np.dot(power_fromgrid, grid_price[starting_hour:ending_hour]))
print(sum(power_fromPV))
plt.show()