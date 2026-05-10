import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyparsing import line


# Constants
G = 6.7e-11 # Gravitation constant 
sog = 5e7 #scale of graph (5e7 means 50 million meters or 50,000 km)
mEarth = 6e24
mCraft = 1e4 # 10 metric tons
rEarth = 6.4e6 # Earth radius in meters
h = 400e3 # 400 km circular orbit
mMoon = 7.35e22 # Moon mass
rMoon = 1.74e6 # Moon radius
dEarthMoon = 384.4e6 # Distance from Earth to Moon

t = 0 # Time in seconds
deltat = 1 # time step in seconds
posEarth = np.array([0,0]) # Earth position
vCraft = np.array([0,7.66e3]) # Orbital Velocity in m/s ()
posCraft = np.array([rEarth + h, 0]) # Initial position of craft
thrust = np.array([0.,0.8]) # Thrust in m/s^2
posMoon = np.array ([dEarthMoon,0])
vMoon = np.array([0,1.022e3])


# init axes    
fig, ax=plt.subplots()
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_xlim(-sog, sog)
ax.set_ylim(-sog, sog)
lineE = ax.plot(posEarth[0], posEarth[1], 'bo')[0]
lineSC = ax.plot(posCraft[0], posCraft[1], 'ro')[0]
 #moon = plt.Circle((posMoon[0], posMoon[1]), rMoon, color='gray', fill=True)

def update(frame):
    global posCraft, vCraft
    
    earth = plt.Circle((0, 0), rEarth, color='blue', fill=True)
   
    r = posCraft - posEarth
    rMag = np.linalg.norm(r)
    rhat = r / rMag
    magF = (G * mEarth * mCraft) / (rMag ** 2)
    gF = -magF * rhat
    acceleration = gF / mCraft
    if t < 3600 or t > 7200:
        vCraft = vCraft + thrust
    vCraft = vCraft + acceleration * deltat
    posCraft = posCraft + vCraft * deltat
    elapsedtimeh = frame // 3600
    elapsedtimem = (frame % 3600) // 60
    elapsedtimes = frame % 6
    pMag = np.linalg.norm(posCraft)
    if pMag < rEarth:
        print("Craft has crashed into Earth!" "Time of crash: " + str(elapsedtimeh) + " hours, " + str(elapsedtimem) + " minutes, " + str(elapsedtimes) + " seconds")
        exit()
    ax.set_aspect('equal')
    lineSC.set_xdata([posCraft[0]])
    lineSC.set_ydata([posCraft[1]])
    ax.add_patch(earth)
    
    r = posMoon - posEarth
    rMag = np.linalg.norm(r)
    rhat = r / rMag
    magF = (G * mEarth * mMoon) / (rMag ** 2)
    gF = -magF * rhat
    acceleration = gF / mMoon
    vMoon = vMoon + acceleration * deltat
    posMoon = posMoon + vMoon * deltat



    return lineSC,

ani = animation.FuncAnimation(
    fig, update,
    frames=range(0, 10801, deltat),
    blit=True, interval=0.1
)


plt.show()