import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


result = pd.read_csv('MinGrid_8MW.csv', index_col = [0], skipfooter =1)
print(result.head())
print(result.tail())
Volume = 84.9492
Area = 15.3649
D =  4.4230
H =5.5288
size = 30
rin = D/2


# 1 Dollar = 0.93 Euro
#Rocks, HT Insulation, LT Insulation, Steel, Foundation
R_ins1 = 0.05
R_steel = 0.02
R_ins2 = 0.25
R2 = rin + R_ins1
V_in1 = np.pi*H * (R2**2 - rin**2)
R3 = R2+R_steel
V_steel = np.pi*H*(R3**2 - R2**2)
R4 = R3 + R_ins2
V_ins2 = np.pi*H*(R4**2 - R3**2)
print(V_in1, V_steel)

CAPEX_Li = 564*0.93*1e3   #Euro/MWh
Size_Li = 8.4 #MWh
CAPEX_battery = CAPEX_Li*Size_Li
#CAPEX_battery = 0.93 * (66 * Volume + 4269 * V_in1 + 616 * V_ins2 + 42354 * V_steel + 1210 * Area + 12000)
print('CAPEX is',CAPEX_battery)

OPEX_cycle = 35  # Euro/MWh
#OPEX_battery = 0.02 * CAPEX_battery + OPEX_cycle * np.sum(result['From Battery'])
OPEX_battery = 35 *0.93 *1.4e3   #Euro/kW-year
lifetime = 30
OPEX_sum = 0
energy_sum = 0
r = 0.07
charge_efficiency = 0.6747
discharge_efficiency = 0.2812

for i in range(1, lifetime + 1):
    OPEX_sum += OPEX_battery / (1 + r) ** i
    energy_sum += np.sum(result['To Battery']*charge_efficiency) / (1 + r) ** i


CAPEX_heater = 70000  # Euro/MW
heater_size = 5.4  # in MW

CAPEX_pipe = 881    # Euro/MWh


LCOS_heat = (CAPEX_battery + CAPEX_heater*heater_size + CAPEX_pipe*size + OPEX_sum) / energy_sum
print('The Heat LCOS is', LCOS_heat)




CAPEX_HEX = 10083  # Euro/MW
HEX_size = 5  # MW


Wind_capacity = 4  # in MW
CAPEX_wind = 1.3e6  # Euro/MW
OPEX_wind = 14.5  # Euro/MWh

PV_capacity = 4  # in MW
CAPEX_PV = 1.01 * 0.93 * 1e6  # Euro/MW
OPEX_PV = 17 * 0.93 # Euro/kW-Year

CAPEX_Li = 564*0.93*1e3   #Euro/MWh
Size_Li = 8.4 #MWh
#Total_CAPEX = CAPEX_wind * Wind_capacity + CAPEX_PV * PV_capacity + CAPEX_heater * heater_size + CAPEX_pipe * size + CAPEX_HEX * HEX_size + CAPEX_battery

Total_CAPEX = CAPEX_wind * Wind_capacity + CAPEX_PV * PV_capacity + 2*CAPEX_battery

Total_OPEX = OPEX_wind * np.sum(result['Wind']) + OPEX_PV*PV_capacity*1000 + np.sum(result['From Grid'] * result['Grid Price']) + OPEX_battery


print('Total CAPEX is',Total_CAPEX)
print('Total OPEX is',Total_OPEX)
energy_sum = 0
OPEX_sum = 0
for i in range(1, lifetime + 1):
    OPEX_sum += Total_OPEX / (1 + r) ** i
    energy_sum += np.sum(result['Load Profile'] + result['To Grid']) / (1 + r) ** i

LCOE = (Total_CAPEX + OPEX_sum) / energy_sum
print('The LCOE is', LCOE)
sys_income = 0

#print(result.loc[result['From Grid'] == 0])

sys_income = np.sum(result['To Grid'] + result['Load Profile'])*LCOE
payback = Total_CAPEX/sys_income


profit = sys_income-Total_OPEX
print('System Income is', sys_income)
print('Payback Period is', payback)
print('Annual Profit is', profit)

#storage_efficiency = np.sum(result['From Battery'])/np.sum(result['To Battery'])
#print('Storage Efficiency is', storage_efficiency)

battery_share = np.sum(result['From Battery'] /np.sum(result['Load Profile'] + result['To Grid'] + result['To Battery']))
print('Battery Share is', battery_share*100, '%')


grid_share = np.average(np.sum(result['From Grid'])/np.sum(result['Load Profile']+ result['To Grid']+result['To Battery']))
print('Grid Share is', grid_share * 100, '%')


renewable_share = np.sum((result['PV'] + result['Wind']) /np.sum(result['Load Profile'] + result['To Grid'] + result['To Battery']))
print('Renewable Penetration is', renewable_share*100)

n_days = 364
battery_utilization = []
for i in range(n_days):
    starting_hour = i*24
    ending_hour = (i+1)*24
    temp1 = np.max((result[starting_hour:ending_hour]['SOC'])) - np.min((result[starting_hour:ending_hour]['SOC']))
    battery_utilization.append(temp1/0.6)

plt.plot(battery_utilization)

print('The average battery utilization is {} %'.format(np.average(battery_utilization)*100))
#plt.show()
#.apply(lambda x: (x-x.iloc[0])/x.iloc[0]*100).reset_index(0, drop=True)
