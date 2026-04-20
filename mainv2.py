# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:53:10 2026

@author: Johann
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

######Constants#########
#earth
G = 6.7e-11
mEarth = 6e24
mCraft = 1e4
rEarth = 6.4e6
h = 400e3

#moon
mMoon = 7.35e22
rMoon = 1.737e6
d = 384.4e6

#other shit
t = 0
deltat = 1
posEarth = np.array([0.0, 0.0])
vCraft = np.array([0.0, 7.66e3])
posCraft = np.array([rEarth + h, 0.0])
posMoon = np.array([d, 0.0])
vMoon = np.array([0.0, 1.022e3])
########################



######LIST FOR TRAIL####
craft_x = [posCraft[0]]
craft_y = [posCraft[1]]

moon_x = [posMoon[0]]
moon_y = [posMoon[1]]
########################



######initial axis######
fig, ax = plt.subplots()
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_xlim(-5e8, 5e8)
ax.set_ylim(-5e8, 5e8)
ax.set_aspect('equal')

lineE = ax.plot(posEarth[0], posEarth[1], 'bo')[0]
lineSC = ax.plot(posCraft[0], posCraft[1], 'ro')[0]
lineM = ax.plot(posMoon[0], posMoon[1], 'go')[0]
########################



######radii for earth and moon######
earthCirc = plt.Circle((0, 0), rEarth, color='blue', fill=True)
moonCirc = plt.Circle((posMoon[0], posMoon[1]), rMoon, color='gray', fill=True)
ax.add_patch(earthCirc)
ax.add_patch(moonCirc)

######trail line code for marks on graph######
trailSC = ax.plot(craft_x, craft_y, 'r-', lw=1)[0]
trailMoon = ax.plot(moon_x, moon_y, 'g-', lw=1)[0]



def checktime(frame):
    elapsedtimeh = frame // 3600
    elapsedtimem = (frame % 3600) // 60
    elapsedtimes = frame % 60
    return elapsedtimeh, elapsedtimem, elapsedtimes

def update(frame):
    global posCraft, vCraft, posMoon, vMoon

    elapsedtimeh, elapsedtimem, elapsedtimes = checktime(frame)
    
    r = posCraft - posEarth
    rMag = np.linalg.norm(r)
    rhat = r / rMag
    magF = (G * mEarth * mCraft) / (rMag ** 2)
    gF = -magF * rhat
    acceleration = gF / mCraft

    if 3600 < frame < 7200:
        vCraft = vCraft + np.array([0.0, 0.4])

    vCraft = vCraft + acceleration * deltat
    posCraft = posCraft + vCraft * deltat
    pMag = np.linalg.norm(posCraft)
        
    r = posMoon - posEarth
    rMag = np.linalg.norm(r)
    rhat = r / rMag
    magF = (G * mEarth * mMoon) / (rMag ** 2)
    gF = -magF * rhat
    acceleration = gF / mMoon
    vMoon = vMoon + acceleration * deltat
    posMoon = posMoon + vMoon * deltat
    
    #####TO update SC TRAIL#####################
    craft_x.append(posCraft[0])
    craft_y.append(posCraft[1])
        #make into a list
    lineSC.set_xdata([posCraft[0]])
    lineSC.set_ydata([posCraft[1]])
        #updates to the new craft position
    trailSC.set_xdata(craft_x)
    trailSC.set_ydata(craft_y)
        #makes the updates for the trail of the spacecraft 
    ############################################  
    
    if pMag < rEarth:
        print(
            "Craft has crashed into Earth! Time of crash: "
            + str(elapsedtimeh) + " hours, "
            + str(elapsedtimem) + " minutes, "
            + str(elapsedtimes) + " seconds"
        )
        ani.event_source.stop()

    return lineSC, lineM, trailSC, trailMoon, moonCirc

ani = animation.FuncAnimation(
    fig, update,
    frames=range(0, 10801, deltat),
    blit=True, interval=20
)

plt.show()
