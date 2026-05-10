import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
G = 6.7e-11 # Gravitation constant 
mEarth = 6e24 #Mass of earth
mCraft = 1e4 # 10 metric tons
rEarth = 6.4e6 # Earth radius in meters
h = 4e5 # 400 km circular orbit
massPayload = 1e4 #mass of payload in kg
massFuel = 26298.85 # mass of fuel in kg
wetMass =  massPayload + massFuel
isp = 340. #specific impulse of rocket
burntime = 60000. #seconds
mdot = massFuel / burntime
exhaustv = 9.81 * isp
thetaM0 = pi
#moon
mMoon = 7.35e22 #Mass Moon kg
rMoon = 1.737e6 #Moon radius km
d = 384.4e6 #km distance Earth to Moon
#thrust
t = 0 # Time in seconds
deltat = 1 # time step in seconds
tstart = 4070
burntimediff = 10
tend = tstart + burntimediff



posEarth = np.array([0,0]) # Earth position
posMoon = np.array ([d,0])
posCraft = np.array([rEarth + h, 0]) # Initial position of craft
vCraft = np.array([0,7.66e3]) # Orbital Velocity in m/s ()

muE = 3.98600436e14 # m3/s2 
muM = 4.903e12 #m3/s2 

tf = 2e6

t = np.linspace(0,tf,10000)

def moon_position(t,posMoon,vMoon):
    r = posMoon - posEarth
    rMag = np.linalg.norm(r)
    rhat = r / rMag
    magF = (G * mEarth * mMoon) / (rMag ** 2)
    gF = -magF * rhat
    acceleration = gF / mMoon
    vMoon = vMoon + acceleration * deltat
    posMoon = posMoon + vMoon * deltat
    return posMoon, vMoon



def df2(x, t):
    r_moon = moon_position(t)              # vector from Earth to Moon at time t
    r_em = x                               # Earth → craft
    r_mc = x - r_moon                      # Moon → craft
    a_earth = -muE * r_em / np.linalg.norm(r_em)**3
    a_moon  = -muM * r_mc / np.linalg.norm(r_mc)**3
    return a_earth + a_moon
def VelocityVerlet(t,x0,v0):
    dt = t[1]-t[0]
    x = np.zeros((len(t),2))
    v = np.zeros((len(t),2))
    
    x[0] = x0
    v[0] = v0
    
    for i in range(len(t)-1):
        vmid = v[i] + 0.5*dt*df2(x[i],t[i])
        x[i+1] = x[i] + dt*vmid
        v[i+1] = vmid + 0.5*dt*df2(x[i+1],t[i+1])
    return x,v


xVV, vVV = VelocityVerlet(t,posCraft, vCraft)


fig1, ax1 = plt.subplots()
ax1.plot(0, 0, 'bo', label="Earth")
ax1.plot(posMoon[:, 0], posMoon[:, 1], 'k--', label="Moon's orbital path")
ax1.plot(xVV[:, 0], xVV[:, 1], label="Velocity Verlet")
ax1.set_aspect("equal")
ax1.set_xlabel("X Position(x - nondimensional)")
ax1.set_ylabel("Y Position(y - nondimensional)")
ax1.legend()
ax1.grid(True)

plt.show()