#for multi-point force reconstruction
import pandas as pd

data = pd.read_csv('single.csv')

data['force1'] = data['intensity1']*0.0000062+(data['intensity2']+data['intensity3'])*0.00000284+0.7489
data['force2'] = data['intensity2']*0.0000062+(data['intensity3']+data['intensity1'])*0.00000284+0.7489
data['force3'] = data['intensity3']*0.0000062+(data['intensity2']+data['intensity1'])*0.00000284+0.7489
#data['force4'] = data['intensity4']*0.0000062+(data['intensity2']+data['intensity3']+data['intensity1'])*0.00000284+0.7489

data.to_csv('single.csv', index=False)

print(data.head())


"""
#for two point force
import pandas as pd
data = pd.read_csv('prediction2_twolength.csv')

data['force1'] = data['intensity1']*0.0000062+data['intensity2']*0.00000284+0.7489
data['force2'] = data['intensity2']*0.0000062+data['intensity1']*0.00000284+0.7489

data.to_csv('prediction2_twolength.csv', index=False)

print(data.head())

"""


