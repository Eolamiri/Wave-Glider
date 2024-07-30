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

def extract_data(wrf_file, lat_point, lon_point):
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

    wind_speed_point = convert_wind_speed(wind_speed_point_10m, 1)
    temp_point = convert_temperature(temp_2m, 1)
    pressure_point = convert_pressure(pressure_2m, temp_2m, 1)
    wind_direction_point = calculate_wind_direction(u_point, v_point)

    data_dict = {
        'Time': times,
        'Wind Speed (m/s)': wind_speed_point,
        'Wind Direction (degrees)': wind_direction_point,
        'Pressure (mb)': pressure_point,
        'Air Temperature (C)': temp_point - 273.15  # Convert to Celsius
    }

    df = pd.DataFrame(data_dict)
    return df

if __name__ == "__main__":
    wrf_file = r"F:\Eolamiri\Output\9th Run (storm Case 2) rerun\wrfout_d03_2022-11-25_12%3A00%3A00"
    points_csv = 'Racetrack_points.csv'
    case_study_name = input("Enter the case study name for the output file: ")
    output_excel = f'extracted_data_{case_study_name}.xlsx'

    # Read the points data
    points_df = pd.read_csv(points_csv)

    # Create an Excel writer
    with pd.ExcelWriter(output_excel) as writer:
        for idx, row in points_df.iterrows():
            lat_point = row['Latitude']
            lon_point = row['Longitude']
            point_id = row['ID']
            
            # Extract data for the current point
            df = extract_data(wrf_file, lat_point, lon_point)
            
            # Write to a new sheet in the Excel file
            df.to_excel(writer, sheet_name=str(point_id), index=False)
    
    print(f"Data successfully extracted to {output_excel}")

