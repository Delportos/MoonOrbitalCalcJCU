import numpy as np
import matplotlib.pyplot as plt
from trajectory import CraftAccel, moonAccel

# Constants
G = 6.7e-11
mEarth = 6e24
rEarth = 6.4e6
h = 4e5
massPayload = 1e4
massFuel = 26298.85
wetMass = massPayload + massFuel
isp = 340.
burntime = 600.
mdot = massFuel / burntime
exhaustv = 9.81 * isp
thrustmag = exhaustv * mdot

mMoon = 7.35e22
rMoon = 1.737e6
d = 384.4e6


# Initial state
posCraft = np.array([rEarth + h, 0.0])
vCraft   = np.array([0.0, 7.66e3])
posMoon  = np.array([d, 0.0])
vMoon    = np.array([0.0, 1.022e3])
mCraft   = wetMass
posEarth = np.array([0.0, 0.0])


def TrajecOptimizerBurn1(startburn1,deltab1,posCraft,posEarth,posMoon,vCraft,vMoon,mCraft,dt,t_total):
    print(f"Running Burn 1 Optimizer for {startburn1}")
    mdot_l = mdot
    thrustmag_l = thrustmag
    n_steps = int(t_total / dt)
    # Preallocated history arrays
    craft_pos = np.zeros((n_steps, 2))
    moon_pos  = np.zeros((n_steps, 2))
    craft_mass = np.zeros(n_steps)
    craft_vel = np.zeros((n_steps, 2))
    times = np.arange(n_steps) * dt
    dscm = np.zeros((n_steps,1))

    crashed = False
    crash_step = None
    startmass = mCraft
    
    for i in range(n_steps):
        tstart = startburn1
        tend = startburn1 + deltab1
        t=i * dt
        a_craft = CraftAccel(posCraft,posEarth,posMoon)
        a_moon = moonAccel(posMoon,posEarth)

        vCraftmid = vCraft + 0.5*a_craft*dt
        vMoonmid = vMoon + 0.5*a_moon*dt

        posCraft = posCraft + vCraftmid * dt
        posMoon = posMoon + vMoonmid * dt

        a_craft = CraftAccel(posCraft,posEarth,posMoon)
        a_moon = moonAccel(posMoon,posEarth)

        vCraft = vCraftmid + 0.5*a_craft * dt
        vMoon = vMoonmid + 0.5*a_moon *dt

        if tstart < t < tend:
            vhat = vCraft / np.linalg.norm(vCraft)
            vCraft = vCraft + vhat * thrustmag_l/mCraft * dt
            mCraft -= mdot_l*dt
            if mCraft <= massPayload:
                mdot_l = 0
                thrustmag_l = 0
        
        craft_pos[i] = posCraft
        moon_pos[i] = posMoon
        craft_mass[i] = mCraft
        craft_vel[i] = vCraft
        dscm[i] = np.linalg.norm(posCraft - posMoon)

        #Crash Checks
        if np.linalg.norm(posCraft) < rEarth + 1e5:
            print(f"Crashed at t={t}s")
            crashed = True
            crash_step = 1
            break
        if np.linalg.norm(dscm) < rMoon:
            print(f"Crashed into at t={t}s")
            crashed = True
            crash_step = 1
            break

    if crashed:
        craft_pos = craft_pos[:crash_step+1]
        moon_pos = moon_pos[:crash_step+1]
        craft_mass = craft_mass[:crash_step+1]
        times = times[:crash_step+1]

    #deltav calculation
    
    mf = mCraft
    deltavburn = exhaustv * np.log(startmass / mf) 

    mass = mCraft
    
    lclosestapproach = np.min(dscm)
    tclosestapproach = np.argmin(dscm)
    print(f"Dv:{deltavburn}")
    print(f"Mass:{mass}")
    print(f"Closest lunar approach: {round((np.min(dscm-rMoon)/1e3),2)} km @ t+{tclosestapproach}s")

    return deltavburn, mass,lclosestapproach, tclosestapproach


def TrajecOptimizerBurn2(startburn1,deltab1,startburn2, deltab2,posCraft,posEarth,posMoon,vCraft,vMoon,mCraft,dt,t_total):
    print(f"Running Burn 2 Optimizer for {startburn2}")
    mdot_l = mdot
    thrustmag_l = thrustmag
    n_steps = int(t_total / dt)
    # Preallocated history arrays
    craft_pos = np.zeros((n_steps, 2))
    moon_pos  = np.zeros((n_steps, 2))
    craft_mass = np.zeros(n_steps)
    craft_vel = np.zeros((n_steps, 2))
    times = np.arange(n_steps) * dt
    dscm = np.zeros((n_steps,1))
    orbitaldscm = np.zeros((n_steps,1))

    crashed = False
    crash_step = None
    startmass = mCraft
    
    for i in range(n_steps):
        tstart = startburn1
        tend = startburn1 + deltab1
        tstart2 = startburn2
        tend2 = startburn2 + deltab2
        t=i * dt
        a_craft = CraftAccel(posCraft,posEarth,posMoon)
        a_moon = moonAccel(posMoon,posEarth)

        vCraftmid = vCraft + 0.5*a_craft*dt
        vMoonmid = vMoon + 0.5*a_moon*dt

        posCraft = posCraft + vCraftmid * dt
        posMoon = posMoon + vMoonmid * dt

        a_craft = CraftAccel(posCraft,posEarth,posMoon)
        a_moon = moonAccel(posMoon,posEarth)

        vCraft = vCraftmid + 0.5*a_craft * dt
        vMoon = vMoonmid + 0.5*a_moon *dt

        if tstart < t < tend:
            vhat = vCraft / np.linalg.norm(vCraft)
            vCraft = vCraft + vhat * thrustmag_l/mCraft * dt
            mCraft -= mdot_l*dt
            if mCraft <= massPayload:
                mdot_l = 0
                thrustmag_l = 0

        if tstart2 < t < tend2:
            vhat = vCraft / np.linalg.norm(vCraft)
            vCraft = vCraft - vhat * thrustmag_l/mCraft * dt
            mCraft -= mdot_l * dt
            if mCraft <= massPayload:
                mdot_l = 0
                thrustmag_l = 0

        craft_pos[i] = posCraft
        moon_pos[i] = posMoon
        craft_mass[i] = mCraft
        craft_vel[i] = vCraft
        dscm[i] = np.linalg.norm(posCraft - posMoon)
        if t < startburn2+deltab2:
            orbitaldscm[i] = -1
        else:
            orbitaldscm[i] = np.linalg.norm(posCraft - posMoon)

        #Crash Checks
        if np.linalg.norm(posCraft) < rEarth + 1e5:
            print(f"Crashed at t={t}s")
            crashed = True
            crash_step = 1
            break
        if np.linalg.norm(dscm) < rMoon:
            print(f"Crashed into at t={t}s")
            crashed = True
            crash_step = 1
            break

    if crashed:
        craft_pos = craft_pos[:crash_step+1]
        moon_pos = moon_pos[:crash_step+1]
        craft_mass = craft_mass[:crash_step+1]
        times = times[:crash_step+1]

    #deltav calculation
    
    mf = mCraft
    deltavburn = exhaustv * np.log(startmass / mf) 

    mass = mCraft
    post_burn = orbitaldscm[orbitaldscm > 0]
    perilune = np.min(post_burn - rMoon) / 1e3
    apolune  = np.max(post_burn - rMoon) / 1e3
    lclosestapproach = np.min(dscm)
    tclosestapproach = np.argmin(dscm)
    print(f"Dv:{deltavburn}")
    print(f"Mass:{mass}")
    print(f"Closest lunar approach: {round((np.min(dscm-rMoon)/1e3),2)} km @ t+{tclosestapproach}s")

    return deltavburn, mass,apolune,perilune

"""def TrajecOptimizerBurn2(startburn1,deltab1,startburn2,deltab2,posCraft,posMoon,vCraft,vMoon,mCraft,trange):

    return deltavburn,mass,apolune"""


#Test 1
print("Test Started!")
tst = 3100
tstarttrials = [tst-5,tst,tst+5,tst+10,tst+20,tst+30,tst+40]
deltab1 = 510
#t1deltatrials = [520,530]
deltavburns =[]
masses = []
lclosestapproaches = []
tclosestapproaches = []

for t in tstarttrials:
    deltav,mass,lclosestapproach,tclosestapproach = TrajecOptimizerBurn1(t,deltab1,posCraft,posEarth,posMoon,vCraft,vMoon,mCraft,1,2.5e5)
    deltavburns.append(deltav)
    masses.append(mass)
    lclosestapproaches.append(lclosestapproach)
    tclosestapproaches.append(tclosestapproach)

#best perilune dscm thats lowest 

target = rMoon + 600e3
closestapproachidx = min(range(len(lclosestapproaches)), 
                         key=lambda i: abs(lclosestapproaches[i] - target))


deltab2 = 90
besttime = tstarttrials[closestapproachidx]
tst2 =  tclosestapproaches[closestapproachidx] - deltab2 / 2
t2starttrial = [tst2-3,tst2-2,tst2-1,tst2,tst2+1,tst2+2,tst2+3]
print(f"Selected t+{besttime} as start time for burn1, with projected perilune of: {round((lclosestapproaches[closestapproachidx]-rMoon)/1e3, 2)} km")

t2starttrial = [tst2-30,tst2-20,tst2-10,tst2,tst2+10,tst2+20,tst2+30]

for t in t2starttrial:
    deltav,mass,apolune,perilune = TrajecOptimizerBurn2(besttime,deltab1,t,deltab2,posCraft,posEarth,posMoon,vCraft,vMoon,mCraft,1,2.5e5)
    deltavburns.append(deltav)
    masses.append(mass)
    print(f"Orbit: Ap:{apolune} km, Pe: {perilune} ")


"""for td in t1deltatrials:
    deltav,mass,lclosestapproach,tclosestapproach = TrajecOptimizerBurn1(t,td,posCraft,posEarth,posMoon,vCraft,vMoon,mCraft,1,4e5)
    deltavburns.append(deltav)
    masses.append(mass)
    lclosestapproaches.append(lclosestapproach)
    tclosestapproaches.append(tclosestapproach)
"""
print(deltavburns)
print(masses)
print(lclosestapproaches)
print(tclosestapproaches)
