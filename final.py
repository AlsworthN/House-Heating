import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression


# getting the outdoor temperature data for kingston upon thames
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 51.406696,
    "longitude": -0.288789,
    "start_date": "2025-04-21",
    "end_date": "2025-04-21",
    "hourly": "temperature_2m",
}
responses = openmeteo.weather_api(url, params = params)

response = responses[0]
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

hourly_data = {
    "date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )
}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_dataframe = pd.DataFrame(data = hourly_data)


# reading the indoor temperature data from the house sensor
df = pd.read_csv(
    "1644725-temp.csv",
    header=None,
    names=["timestamp", "indoor_temp"])

# changing the unix timestamps into normal dates and times
df["time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

# the sensor records loads of values so this gets one average temperature for each hour
hourly_indoor = (
    df.set_index("time")["indoor_temp"]
      .resample("1h")
      .mean())

# using the same day as the outdoor temperature data
day = "2025-04-21"
indoor_day = hourly_indoor.loc[day]


# comparing the exact equation with eulers method so i can see the approximation error
T0 = 20
Tout = 10
k = 0.1
dt = 1
hours = 24

times = []
temps = []

# exact newtons law of cooling equation
for t in range(0,hours):
    temp = Tout + (T0 - Tout)*math.e**(-k*t)
    times.append(t)
    temps.append(temp)

# eulers method for the same situation
temps2 = [T0]
for t in range(0,hours):
    T = temps2[t]
    newtemp = T + (dt*(-1*k*((T - Tout))))
    temps2.append(newtemp)

temps2.pop()

# putting both lines on the same graph so they are easier to compare
plt.plot(times, temps, label = "Actual", color = "red")
plt.plot(times, temps2, label = "Euler's Approx.", color = "blue")
plt.xlabel("Time / hours")
plt.ylabel("Temperature / C")
plt.legend()
plt.show()

# getting the error at each hour between the exact value and eulers approximation
error = []
for y in range(len(temps)):
    error_at_time = temps[y] - temps2[y]
    error.append(error_at_time)


# finding one constant k value from the real indoor and outdoor temperatures
model = LinearRegression()
x = []
y = []
dt = 1

for j in range(len(indoor_day)-1):
    x.append(indoor_day.values[j] - hourly_temperature_2m[j])
    next_indoor = indoor_day.values[j+1]
    y.append((next_indoor - indoor_day.values[j])/dt)

model.fit(
    np.array(x).reshape(-1,1),
    y
)

k = -model.coef_[0]
print("Estimated k:", k)



# 1 means someone is in the house and 0 means nobody is there
occupied = [1,0,0,1,0,1,0,0,0,1,1,1,0,1,1,1,0,0,1,1,0,1,0,1]
comfort = []

# using a lower minimum temperature when nobody is in the house
for i in occupied:
    if i == 0:
        comfort.append(14)
    if i == 1:
        comfort.append(18)


# running the heating model using eulers method
h = 1.23 # arbitary value used for how much the temperature is icnreased each timestep by the radiator
T0 = 15 # arbitary starting indoor temp used
Tout = hourly_temperature_2m
H_at_time = []
dt = 1
hours = 24
times = []
temps3 = [T0]

for t in range(0,hours):
    times.append(t)
    T = temps3[t]

    # first finding what the indoor temperature would be without the heating
    pure_indoor = (T + (dt*(-1*k*((T - Tout[t]))))) + error[t]

    # if it goes below the comfort temperature the heating turns on
    if pure_indoor <= comfort[t]:
        H = 1
        H_at_time.append(1)
    else:
        H = 0
        H_at_time.append(0)

    # adding the heating effect if the heating is on
    newtemp = pure_indoor + (h*H) + error[t]
    temps3.append(newtemp)


# reading the carbon intensity csv and only using london because kingston is in london
carbon_intensity = pd.read_csv("regional_carbon_intensity.csv")
carbon_intensity["datetime"] = pd.to_datetime(carbon_intensity["datetime"])

london = carbon_intensity[["datetime", "London"]]

# the carbon data is every 30 mins so this turns it into one average for each hour
carbon_hourly = (
    carbon_intensity
    .set_index("datetime")["London"]
    .resample("1h")
    .mean()
)

# getting the last 24 hours and making it a list so it is easier to use
last24_carbon_hourly = carbon_hourly.tail(24).tolist()


# working out the energy used and carbon produced each hour
P = 2 # heater power in kW
dt = 1 # time in hours

energy_per_hour = []
carbon_per_hour = []

for i in range(24):
    energy = P*H_at_time[i]*dt
    carbon = energy*last24_carbon_hourly[i]

    energy_per_hour.append(energy)
    carbon_per_hour.append(carbon)

# adding values for each hour together to get the totals for the whole day
total_energy = sum(energy_per_hour)
total_carbon = sum(carbon_per_hour)

# final graph to show temperature at house versus the comfort threshold
adjustedtimes = list(range(25))
plt.plot(adjustedtimes,temps3,label = "Indoor temperature")
plt.plot(times,comfort,label = "Comfort Threshold temperature")

plt.xlabel("Time / hours")
plt.ylabel("Temperature / °C")

plt.legend()
plt.show()

print("Heating on/off each hour:", H_at_time)
print("Total energy used:", total_energy, "kWh")
print("Total carbon emissions:", total_carbon/1000, "kg CO2")
