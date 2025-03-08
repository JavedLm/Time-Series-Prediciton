import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path1= 'C:/Users/Javed Khan/Downloads/TV02_Bosch/AOI_Data.csv' 
file_path2= 'C:/Users/Javed Khan/Downloads/TV02_Bosch/SPI_Data.csv'
file_path3= 'C:/Users/Javed Khan/Downloads/TV02_Bosch/AXI_Data.csv'
file_path4= 'C:/Users/Javed Khan/Downloads/TV02_Bosch/RUL.csv'

# Read the CSV file

aoi_data = pd.read_csv('file_path1')
axi_data = pd.read_csv('file_path2')
rul_data = pd.read_csv('file_path3')
spi_data = pd.read_csv('file_path4')

# Merge AOI, AXI, and SPI with RuL

# Merge static data on unit_id

static_data = pd.merge(aoi_data, spi_data, on='unit_id', how='inner')
static_data = pd.merge(static_data, axi_data, on='unit_id', how='inner')

# Add static features to time-series data
merged_data = pd.merge(rul_data, static_data, on='unit_id', how='left')

# Sort by unit_id and time
merged_data = merged_data.sort_values(['unit_id', 'timestamp'])