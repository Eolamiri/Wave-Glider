import netCDF4 as nc
import pandas as pd
import numpy as np

def find_nearest_grid_point(lats, lons, lat_point, lon_point):
    distance = np.sqrt((lats - lat_point)**2 + (lons - lon_point)**2)
    min_idx = np.unravel_index(np.argmin(distance, axis=None), distance.shape)
    return min_idx

def calculate_wind_speed(u, v):
    return np.sqrt(u**2 + v**2)

def calculate_wind_direction(u, v):
    wind_dir_rad = np.arctan2(v, u)
    wind_dir_deg = np.degrees(wind_dir_rad)
    wind_dir_deg = (wind_dir_deg + 360) % 360
    return wind_dir_deg

def convert_wind_speed(wind_speed_10m, z, zr=10, alpha=0.1):
    return wind_speed_10m * (z / zr) ** alpha

def convert_temperature(temp_2m, z, zr=2, lapse_rate=-6.5):
    # Temperature in Kelvin at height z using lapse rate (in K/km)
    return temp_2m + lapse_rate * (z - zr) / 1000

def convert_pressure(pressure_2m, temp_2m, z, zr=2):
    # Using the barometric formula to convert pressure to height z
    g = 9.80665  # Gravity (m/s^2)
    M = 0.0289644  # Molar mass of Earth's air (kg/mol)
    R = 8.3144598  # Universal gas constant (J/(mol·K))
    return pressure_2m * np.exp(-(g * M * (z - zr)) / (R * temp_2m))

def extract_data(wrf_file, lat_points, lon_points, output_csv):
    dataset = nc.Dataset(wrf_file)

    # Extract the necessary variables
    times = dataset.variables['Times'][:]
    lats = dataset.variables['XLAT'][0, :, :]
    lons = dataset.variables['XLONG'][0, :, :]
    u_wind = dataset.variables['U10'][:]
    v_wind = dataset.variables['V10'][:]
    pressure = dataset.variables['PSFC'][:]
    temperature = dataset.variables['T2'][:]

    times = [''.join(time.tobytes().decode('utf-8').strip()) for time in times]
    times = pd.to_datetime(times, format='%Y-%m-%d_%H:%M:%S')

    data_dict = {
        'Time': times,
        'Wind Speed of buoy': [], 'Wind Direction of buoy': [],
        'Pressure of buoy': [], 'Air Temperature of buoy': [],
        'Wind Speed of WG': [], 'Wind Direction of WG': [],
        'Pressure of WG': [], 'Air Temperature of WG': []
    }

    for i, (lat_point, lon_point) in enumerate(zip(lat_points, lon_points)):
        lat_idx, lon_idx = find_nearest_grid_point(lats, lons, lat_point, lon_point)

        if lat_idx >= lats.shape[0] or lon_idx >= lons.shape[1]:
            raise IndexError("Calculated indices are out of bounds")

        # Extracting 3x3 grid points for averaging
        lat_indices = slice(max(lat_idx-1, 0), min(lat_idx+2, lats.shape[0]))
        lon_indices = slice(max(lon_idx-1, 0), min(lon_idx+2, lons.shape[1]))

        u_point = np.mean(u_wind[:, lat_indices, lon_indices], axis=(1, 2))
        v_point = np.mean(v_wind[:, lat_indices, lon_indices], axis=(1, 2))
        wind_speed_point_10m = calculate_wind_speed(u_point, v_point)

        temp_2m = np.mean(temperature[:, lat_indices, lon_indices], axis=(1, 2))
        pressure_2m = np.mean(pressure[:, lat_indices, lon_indices], axis=(1, 2)) / 100  # Convert to hPa

        if i == 0:
            wind_speed_point = convert_wind_speed(wind_speed_point_10m, 3.4)
            temp_point = convert_temperature(temp_2m, 3.4)
            pressure_point = convert_pressure(pressure_2m, temp_2m, 3.4)
        elif i == 1:
            wind_speed_point = convert_wind_speed(wind_speed_point_10m, 1)
            temp_point = convert_temperature(temp_2m, 1)
            pressure_point = convert_pressure(pressure_2m, temp_2m, 1)

        wind_direction_point = calculate_wind_direction(u_point, v_point)

        if i == 0:
            data_dict['Wind Speed of buoy'] = wind_speed_point
            data_dict['Wind Direction of buoy'] = wind_direction_point
            data_dict['Pressure of buoy'] = pressure_point
            data_dict['Air Temperature of buoy'] = temp_point - 273.15  # Convert to Celsius
        elif i == 1:
            data_dict['Wind Speed of WG'] = wind_speed_point
            data_dict['Wind Direction of WG'] = wind_direction_point
            data_dict['Pressure of WG'] = pressure_point
            data_dict['Air Temperature of WG'] = temp_point - 273.15  # Convert to Celsius

    df = pd.DataFrame(data_dict)
    df.to_csv(output_csv, index=False)
    print(f"Data successfully extracted to {output_csv}")

if __name__ == "__main__":
    wrf_file = r"F:\Eolamiri\Output\9th Run (storm Case 2) rerun\wrfout_d03_2022-11-25_12%3A00%3A00"
    lat_points = [38.460, 38.65812]
    lon_points = [-74.692, -74.55963]
    output_csv = 'extracted_data_STORM_CASE_2 new (3x3 grid).csv'
    
    extract_data(wrf_file, lat_points, lon_points, output_csv)
    
    
