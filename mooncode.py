import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyparsing import line


# Constants
G = 6.7e-11 # Gravitation constant 
mEarth = 6e24 #Mass of earth
mCraft = 1e4 # 10 metric tons
rEarth = 6.4e6 # Earth radius in meters
h = 400e3 # 400 km circular orbit

mMoon = 7.35e22 #Mass Moon kg
rMoon = 1.737e6 #Moon radius km

d = 384.4e6 #km distance Earth to Moon

t = 0 # Time in seconds
deltat = 60 # time step in seconds
posEarth = np.array([0,0]) # Earth position
vCraft = np.array([0,7.66e3]) # Orbital Velocity in m/s ()
posCraft = np.array([rEarth + h, 0]) # Initial position of craft
posMoon = np.array ([d,0])
vMoon = np.array([0,1.022e3])

# init axes    
fig, ax=plt.subplots()
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_xlim(-5e8, 5e8)
ax.set_ylim(-5e8, 5e8)
lineE = ax.plot(posEarth[0], posEarth[1], 'bo')[0]
lineSC = ax.plot(posCraft[0], posCraft[1], 'ro')[0]
lineM = ax.plot(posMoon[0], posMoon[1], 'go')[0]



def update(frame):
    global posCraft, vCraft, posMoon, vMoon

    r = posCraft - posEarth
    rMag = np.linalg.norm(r)
    rhat = r / rMag
    magF = (G * mEarth * mCraft) / (rMag ** 2)
    gF = -magF * rhat
    acceleration = gF / mCraft
    if t < 3600 or t > 7200:
        vCraft = vCraft + np.array([0,0.5])
    vCraft = vCraft + acceleration * deltat
    posCraft = posCraft + vCraft * deltat

    lineSC.set_xdata([posCraft[0]])
    lineSC.set_ydata([posCraft[1]])
    
    r = posMoon - posEarth
    rMag = np.linalg.norm(r)
    rhat = r / rMag
    magF = (G * mEarth * mMoon) / (rMag ** 2)
    gF = -magF * rhat
    acceleration = gF / mMoon
    vMoon = vMoon + acceleration * deltat
    posMoon = posMoon + vMoon * deltat

    lineM.set_xdata([posMoon[0]])
    lineM.set_ydata([posMoon[1]])
    
    return lineSC, lineM,


ani = animation.FuncAnimation(
    fig, update,
    frames=range(0, 10801, deltat),
    blit=True, interval=20
)

plt.show()
