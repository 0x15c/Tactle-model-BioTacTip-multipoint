#for single point orce reconstruction
import pandas as pd

data = pd.read_csv('single.csv')
print(data.columns)


def calculate_force(intensity):
    A = 4e-17
    B = -2.6465e-11
    C = 1.0643e-5
    D = 0.6503
    return A * intensity**3 + B * intensity**2 + C * intensity + D

data['force'] = data['intensity'].apply(calculate_force)
#save data
data.to_csv('single.csv', index=False)

print(data.head())
