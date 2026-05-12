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



"""
This function takes 2 parameters that vary (startburn1 and deltab1), and a number of other parameters to properly calculate gravitational forces and for our approximation
I
"""
def TrajecOptimizerBurn1(startburn1,deltab1,posCraft,posEarth,posMoon,vCraft,vMoon,mCraft,dt,t_total):
    print(f"Running Burn 1 Optimizer for {startburn1}")
    mdot_l = mdot #local variable
    thrustmag_l = thrustmag #local variable
    n_steps = int(t_total / dt) ## of interation steps
    # Preallocated history arrays
    craft_pos = np.zeros((n_steps, 2)) 
    moon_pos  = np.zeros((n_steps, 2))
    craft_mass = np.zeros(n_steps)
    craft_vel = np.zeros((n_steps, 2))
    dscm = np.zeros((n_steps,1))

    #starting mass of our craft
    startmass = mCraft
    
    #iterates through entire trajectory for given time frame
    for i in range(n_steps):
        #burn params and time step calculation
        tstart = startburn1
        tend = startburn1 + deltab1
        t=i * dt
        
        #Velocity Verlet 
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

        #during the burn times, we accelerate along our current vector, dropping mass
        if tstart < t < tend: # if t is during burn times, thrust
            vhat = vCraft / np.linalg.norm(vCraft) #find current velocity vector
            vCraft = vCraft + vhat * thrustmag_l/mCraft * dt #alter velocity vector by thrust from rocket engine
            mCraft -= mdot_l*dt #account for fuel burn by multiplying time step by our mass flow (mdot)
            if mCraft <= massPayload: #if the craft's mass is less than the payload, no fuel left, cut thrust entirely, cut mass flow entirely
                mdot_l = 0 #local variable of constant
                thrustmag_l = 0 #localvaribale of constant
        
        #record: positon of craft,moon, mass, of craft, 
        craft_pos[i] = posCraft #pos craft
        moon_pos[i] = posMoon #moon pos
        craft_mass[i] = mCraft #mass of craft
        craft_vel[i] = vCraft #velocity of craft
        dscm[i] = np.linalg.norm(posCraft - posMoon) #distance betwen craft and center of the moon (rvector between SC and Moon)

        #Crash Checks
        if np.linalg.norm(posCraft) < rEarth + 1e5: #if the position is less than the earth's radius + 100km (atmosphere) we crash
            print(f"Crashed at t={t}s")
            break
        if np.linalg.norm(dscm) < rMoon: #if the rvector is less than the moons radius, we crash
            print(f"Crashed into at t={t}s")
            break
    # exit for loop
    #deltav calculation
    
    mf = mCraft #final mass of our craft
    deltavburn = exhaustv * np.log(startmass / mf)  #total deltav calculation

    mass = mCraft
    
    #print statements
    lclosestapproach = np.min(dscm)
    tclosestapproach = np.argmin(dscm)
    print(f"Dv:{round(deltavburn,3)}")
    print(f"Mass:{round(mass,3)}")
    print(f"Closest lunar approach: {round((np.min(dscm-rMoon)/1e3),2)} km @ t+{tclosestapproach}s")
    print("")

    return deltavburn, mass,lclosestapproach, tclosestapproach 
    #returns the dv, the mass of the craft, the closest approach's distance from the moon and the time
"""
This function takes 4 parameters that vary (startburn1 and deltab1), and a number of other parameters to properly calculate gravitational forces and for our approximation
I
"""
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

    
    startmass = mCraft
    
    for i in range(n_steps):
        #burn params and time step calculation
        tstart = startburn1
        tend = startburn1 + deltab1
        tstart2 = startburn2
        tend2 = startburn2 + deltab2
        t=i * dt

        #Velocity Verlet
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


        #during the burn times, we accelerate along our current vector, drop mass
        if tstart < t < tend:
            vhat = vCraft / np.linalg.norm(vCraft) #velocity direciton
            vCraft = vCraft + vhat * thrustmag_l/mCraft * dt #alter velocity vector by thrust from rocket engine
            mCraft -= mdot_l*dt  #account for fuel burn by multiplying time step by our mass flow (mdot)
            if mCraft <= massPayload: #if the craft's mass is less than the payload, no fuel left, cut thrust entirely, cut mass flow entirely
                mdot_l = 0 #local variable
                thrustmag_l = 0 #local variabel
        #exact same as above, now we just use the 2nd burn parameters instead of the first
        if tstart2 < t < tend2:
            vhat = vCraft / np.linalg.norm(vCraft)
            vCraft = vCraft - vhat * thrustmag_l/mCraft * dt
            mCraft -= mdot_l * dt
            if mCraft <= massPayload:
                mdot_l = 0
                thrustmag_l = 0
            
        #record: positon of craft,moon, mass, of craft, 
        craft_pos[i] = posCraft 
        moon_pos[i] = posMoon
        craft_mass[i] = mCraft
        craft_vel[i] = vCraft
        dscm[i] = np.linalg.norm(posCraft - posMoon) #calculates distance from moon

        #if both burns have not yet occured, the orbital distance between the space craft and moon is set to -1
        if t < startburn2+deltab2: #pre burn
            orbitaldscm[i] = -1
        else: #once both burns have completed, the distance between the spacecraft and moon is recorded 
            orbitaldscm[i] = np.linalg.norm(posCraft - posMoon) # done with all burns, final ORBIT

        #Crash Checks
        if np.linalg.norm(posCraft) < rEarth + 1e5:
            print(f"Crashed at t={t}s")
            break
        if np.linalg.norm(dscm) < rMoon:
            print(f"Crashed into at t={t}s")
            break

    
    #deltav calculation
    
    mf = mCraft
    deltavburn = exhaustv * np.log(startmass / mf) 

    mass = mCraft
    post_burn = orbitaldscm[orbitaldscm > 0] #extracts every value into new list that exceeds 0
    perilune = np.min(post_burn - rMoon) / 1e3 #finds minimum of all values and converts to km (Subtracts rMoon, because that is the moon's surface)
    apolune  = np.max(post_burn - rMoon) / 1e3 #finds max of all values and converts to km
    #lclosestapproach = np.min(dscm)
    tclosestapproach = np.argmin(dscm)
    #print statements
    print(f"Dv:{round(deltavburn,3)}")
    print(f"Mass:{round(mass,3)}")
    print(f"Closest lunar approach: {round((np.min(dscm-rMoon)/1e3),2)} km @ t+{tclosestapproach}s")
    
    return deltavburn,mass,apolune,perilune
#returns the deltav, the mass of the rocket, and the apolune and perilune



#Test 1
print("Test Started!")
tst = 3100 #a test time
# tstarttrials = [tst-5,tst,tst+5,tst+10,tst+20,tst+30,tst+40] First tests to find optimal trajectory
#tstarttrials = [tst+30,tst+31,tst+32,tst+33,tst+34,tst+35] # second test to find optimal trajectory
tstarttrials =[tst+30,tst+32,tst+38] #demo code to demonstrate
deltab1 = 510  #length of first burn

#empty lists to record results of for loop below
deltavburns =[]
masses = []
lclosestapproaches = []
tclosestapproaches = []
apolunes = []
perilunes = []
for t in tstarttrials:
    #passes outputs for TrajectoryBurn and appends them to lists
    deltav,mass,lclosestapproach,tclosestapproach = TrajecOptimizerBurn1(t,deltab1,posCraft,posEarth,posMoon,vCraft,vMoon,mCraft,1,2.5e5)
    deltavburns.append(deltav)
    masses.append(mass)
    lclosestapproaches.append(lclosestapproach)
    tclosestapproaches.append(tclosestapproach)

#best perilune dscm thats lowest 
target = rMoon + 600e3 # target perilune is ~600km above lunar surface to ensure no crashing during 2nd burn (perilune will drop)

#for loop below figures out which of approaches is closest to our desired altitude
closestapproachidx = 0
smallest_diff = abs(lclosestapproaches[0] - target)
for i in range(1, len(lclosestapproaches)):
    diff = abs(lclosestapproaches[i] - target)
    if diff < smallest_diff:
        smallest_diff = diff
        closestapproachidx = i

deltab2 = 90 # time of burn 2
besttime = tstarttrials[closestapproachidx] #start time of the best result
tst2 =  tclosestapproaches[closestapproachidx] - deltab2 / 2 #centers burn on our perilune to optimize fuel usage

print(f"Selected t+{besttime} as start time for Burn 1, with projected perilune of: {round((lclosestapproaches[closestapproachidx]-rMoon)/1e3, 2)} km")
#print statemennt

#t2starttrial = [tst2-30,tst2-20,tst2-10,tst2,tst2+10,tst2+20,tst2+30] #previous tests
t2starttrial = [tst2+20,tst2+30] #demo 

#similar to above
for t in t2starttrial:
    #passes variables from function into lists for analysis
    deltav,mass,apolune,perilune = TrajecOptimizerBurn2(besttime,deltab1,t,deltab2,posCraft,posEarth,posMoon,vCraft,vMoon,mCraft,1,2.5e5)
    deltavburns.append(deltav)
    masses.append(mass)
    apolunes.append(apolune)
    perilunes.append(perilune)
    #print statements
    print(f"DeltaV: {round(deltav,3)}")
    print(f"Orbit: Ap:{round(apolune,3)} km, Pe: {round(perilune,3)} ")
    print("")

#for loop below figures out which of the perilunes is closest to our target
target = 100e3 # 100km desired perilune
closestapproachidxpl = 0
smallest_difference = abs(perilunes[0]-target)

for i in range(1,len(perilunes)):
    diff = abs(perilunes[i]-target)
    if diff < smallest_diff:
        smallest_diff = diff
        closestapproachidxpl = i


#best time for our burn to begin given our burn time
besttime2 = t2starttrial[closestapproachidxpl]

#final print statement to show best values to plug into our trajectory visualizer
print(f"Best Trajectory: \nBurn 1:t+{besttime}s for {deltab1}s\nBurn 2:t+{besttime2}s for {deltab2}s\nAchieved orbit: Ap:{round(apolunes[closestapproachidxpl],2)} km, Pe:{round(perilunes[closestapproachidxpl],2)} km")

"""
Debug
print(deltavburns)
print(masses)
print(lclosestapproaches)
print(tclosestapproaches)
"""
