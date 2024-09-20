import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.font_manager as fm

font_path = r"C:\Users\lenovo\AppData\Local\Microsoft\Windows\Fonts\Helvetica.ttf"
font_prop = fm.FontProperties(fname=font_path)
font_name = font_prop.get_name()
plt.rcParams.update({
    'font.serif': font_name,
    'font.size': 25,
    'axes.titlesize': 25,
    'axes.labelsize': 25,
    'legend.fontsize': 22,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
})

data = pd.read_csv('multi-intensity.csv')
time=data['Time']
force1=data['force1']
force2=data['force2']
force3=data['force3']
force4=data['force4']

fig, ax = plt.subplots(figsize=(15, 8))

#line2, = ax.plot([], [], label='Force 1 Prediction',color=[218/255,84/255,17/255],linewidth=2)
line1, = ax.plot([], [], label='Predicted force1',color=[218/255,84/255,17/255], linestyle='-', linewidth=3)
line2, = ax.plot([], [], label='Predicted force2',color=[0/255,114/255,192/255], linestyle='-', linewidth=3)
line3, = ax.plot([], [], label='Predicted force3',color=[97/255,132/255,100/255], linestyle='-', linewidth=3)
line4, = ax.plot([], [], label='Predicted force4',color=[238/255,176/255,33/255], linestyle='-', linewidth=3)

ax.set_xlim(0, 30)
ax.set_ylim(0.5, 3.5)

#ax.set_title('Force Prediciton')
ax.set_xlabel('Time(s)')
ax.set_ylabel('Force(N)')

#ax.legend(loc='upper left', fontsize='small', ncol=2, bbox_to_anchor=(0.02, 0.18))

def update(frame):
    line1.set_data(time[:frame], force1[:frame])
    line2.set_data(time[:frame], force2[:frame])
    line3.set_data(time[:frame], force3[:frame])
    line4.set_data(time[:frame], force4[:frame])

    return line1,line2,line3,line4
#create animation
ani = FuncAnimation(fig, update, frames=len(time), interval=100, blit=True)

plt.show()

