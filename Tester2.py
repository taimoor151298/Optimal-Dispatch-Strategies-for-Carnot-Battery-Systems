from Weekly_Dispatcher import weekly_dispatcher
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Reading PV power and converting it into W. Size is set to 2MW

df1 = pd.read_csv('PV_power.csv')
PV_power = 50*np.array(df1['System power generated | (kW)']/1000)

print('PV Power Profile Loaded')

### Reading wind power. System size is set to 2MW
df2 = pd.read_csv('Wind_power.csv')
Wind_power = 5*np.array(df2['System power generated | (kW)']/1000)
print('Wind Power Profile Loaded')

### Reading the load profile


df4 = pd.read_excel('2021_ElectricityPrice.xlsx', sheet_name= 'Database', skiprows= range(3))
df4 = df4[['day', 'DE-LU']]
grid_price = np.array(df4[['DE-LU']])
print('Electricity Prices Loaded')

load_profile = np.array(1000*(pd.read_csv('Load.csv', header = None)[0]))
print('Load Profile Loaded')

n_days = 7
n_steps = 21
starting_hour = n_days*(n_steps-1)*24
ending_hour = n_days*n_steps*24
x = np.linspace(starting_hour,ending_hour-1,168)
plt.legend(["PV", "From Battery", "Wind", "From Grid", "To Battery", "To Grid", "Load" ])

plt.step(x,PV_power[starting_hour:ending_hour])
plt.step(x,Wind_power[starting_hour:ending_hour])
plt.step(x,load_profile[starting_hour:ending_hour])

#plt.step(x,grid_price[starting_hour:ending_hour])
#plt.step(x,-power_tobattery[starting_hour:ending_hour])
#plt.step(x,-power_toGrid[starting_hour:ending_hour])
#plt.step(x,load_profile[starting_hour:ending_hour])
plt.legend(["PV", "Wind", "Load" ])
plt.figure()
plt.plot(grid_price[0:ending_hour])
print(np.power(0.95,1/24))
plt.figure()
load_profile = load_profile[0:8736]
avg = np.average(load_profile.reshape(-1, 168), axis=1)
max = np.max(load_profile.reshape(-1, 168), axis=1)
print(sum(load_profile))
y = np.linspace(0, 51,52)
plt.subplot(2,1,1)
plt.step( y, avg)
plt.subplot(2,1,2)
plt.step(y,max)
plt.show()