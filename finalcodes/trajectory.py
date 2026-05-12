import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.7e-11 #Gravitational constant in m^3/(kg*s^2)
mEarth = 6e24 #Mass of Earth in kg
rEarth = 6.4e6 #Radius of Earth in meters
h = 4e5 # height of spacecraft above earth's surface
massPayload = 1e4  #Dry mass of the spacecraft in kg
massFuel = 26298.85 #Mass of fuel in kg
wetMass = massPayload + massFuel #Total starting mass
isp = 340. #Specific impulse of the engine in seconds
burntime = 600. #Total burn time in seconds
mdot = massFuel / burntime  #Fuel mass flow rate in kg/s
exhaustv = 9.81 * isp #Exhaust velocity in m/s 
thrustmag = exhaustv * mdot #Thrust magnitude in N

mMoon = 7.35e22  #Mass of Moon in kg
rMoon = 1.737e6 #Radius of Moon in meters
d = 384.4e6 #Distance of Moon to earth in meters

# Burn windows
t1delta = 510 #Length of first burn in seconds
tstart = 3132 #starting time of first burn in seconds (selected from optimizer)
tend = tstart+t1delta #Ending time of the burn in secodns 

t2delta= 90 #Length of second burn in seconds
tstart2 = 200508 #Starting time of the second burn in seconds(selected from optimizer)
tend2 = tstart2 +t2delta #Ending time of the burn in seconds

# Sim params
dt = 1#Time step in seconds
t_total = 4e5 #Total run time in s 
n_steps = int(t_total / dt)  #Number of steps as an interger

# Initial state
posCraft = np.array([rEarth + h, 0.0]) # Initial position of the spacecraft (400 km above surface of earth)
vCraft   = np.array([0.0, 7.66e3])  #Initial velocity of the spacecraft
posMoon  = np.array([d, 0.0])  #Initial position of the Moon
vMoon    = np.array([0.0, 1.022e3]) #Initial velocity of the Moon
mCraft   = wetMass #Mass of the spacecraft
posEarth = np.array([0.0, 0.0]) #Initial position of the Earth (Assuming it is fixed at (0,0)

# Preallocated history arrays (length is the number of time steps)
craft_pos = np.zeros((n_steps, 2)) #Stores spacecraft position
moon_pos  = np.zeros((n_steps, 2))#Stores Moon position
craft_mass = np.zeros(n_steps) #Stores mass number
craft_vel = np.zeros((n_steps, 2))#Stores velocity of the spacecraft
times = np.arange(n_steps) * dt#Creates an array of time values
dscm = np.zeros((n_steps,1))  #Stores distance between spacecraft and moon



def CraftAccel(posCraft, posEarth,posMoon): #Def function to calculate spacecraft's acceleration
    # Gravity from Earth on craft
    r_e = posCraft - posEarth  #Vector of Earth to spacecraft
    r_e_mag = np.linalg.norm(r_e) #Distance of Earth to spacecraft
    a_earth = -G * mEarth * r_e / r_e_mag**3   # combines rhat and 1/r^2 for Earth's gravity 1 /rmag^2 * r / rmag = r / rmag^3


    # Gravity from Moon on craft
    r_m = posCraft - posMoon #Vector of Moon to Spacecraft
    r_m_mag = np.linalg.norm(r_m)#Distance of Moon to spacecraft
    a_moon = -G * mMoon * r_m / r_m_mag**3 # combines rhat and 1/r^2 for Moons's gravity, 1 /rmag^2 * r / rmag = r / rmag^3

    acceleration = a_earth + a_moon#acc1; Total acceleration of the spacecraft due to Earrth and Moon's gravity
    return acceleration #Returns the spacecraft's acceleration value

def moonAccel(posMoon,posEarth):  #Def function to calculate Moon's acceleration
    r = posMoon - posEarth #Vector of Earth to Moon
    rmag = np.linalg.norm(r)  #Distance of Earth to Moon
    a = -G*mEarth*r / rmag**3 #combines rhat and 1/r^2 for Moon's acceleration, 1 /rmag^2 * r / rmag = r / rmag^3

    return a #returns moon's acceleration value

 #if this file is being directly run, the following runs, if function from file is called elsewhere, python ignores the following
if __name__ == "__main__":

    for i in range(n_steps): #simulation runs for each time step
        t = i * dt  #Adds a tick to the t count

        #velocity verlet
        a_craft = CraftAccel(posCraft,posEarth,posMoon)# Gravity from Earth on craft
        a_moon = moonAccel(posMoon,posEarth)  #Adds a tick to the t count

        vCraftmid = vCraft + 0.5*a_craft*dt #Calculates midpoint velocity for the spacecraft
        vMoonmid = vMoon + 0.5*a_moon*dt#Calculates midpoint velocity for the Moon


        posCraft = posCraft + vCraftmid * dt #Updates the position of the spacecraft
        posMoon = posMoon + vMoonmid * dt #Updates the position of the Moon


        a_craft = CraftAccel(posCraft,posEarth,posMoon)# Gravity from Earth on craft
        a_moon = moonAccel(posMoon,posEarth) #Calculations acceleration on the moon

        vCraft = vCraftmid + 0.5*a_craft * dt  #Updates the velocity of the spacecraft using 2nd accel
        vMoon = vMoonmid + 0.5*a_moon * dt #Updates the velocity of the Moon using 2nd accel

        # Burn 1 (prograde)
        if tstart < t < tend: #Checks if t is inside the parameters of the first burn time
            vhat = vCraft / np.linalg.norm(vCraft) #velocity direciton
            vCraft = vCraft + vhat * thrustmag/mCraft * dt #alter velocity vector by thrust from rocket engine
            mCraft -= mdot * dt #account for fuel burn by multiplying time step by our mass flow (mdot)
            if mCraft <= massPayload: #if the craft's mass is less than the payload, no fuel left, cut thrust entirely, cut mass flow entirely
                mdot = 0
                thrustmag = 0

        # Burn 2 (retrograde) exact same as above, but thrust is in opposite direction to velocity vector to slow down
        if tstart2 < t < tend2:
            vhat = vCraft / np.linalg.norm(vCraft)
            vCraft = vCraft - vhat * thrustmag/mCraft * dt
            mCraft -= mdot * dt
            if mCraft <= massPayload:
                mdot = 0
                thrustmag = 0
         #record: positon of craft,moon, mass,velocity
        craft_pos[i] = posCraft
        moon_pos[i]  = posMoon
        craft_mass[i] = mCraft
        craft_vel[i] = vCraft 
        dscm[i] = np.linalg.norm(posCraft - posMoon) #calculatioes distance fropm moon

        # Crash check
        if np.linalg.norm(posCraft) < rEarth + 1e5: #if the position is less than the earth's radius + 100km (atmosphere) we crash
            print(f"Crashed at t={t}s")
            break
        if np.linalg.norm(dscm) < rMoon: #if the rvector is less than the moons radius, we crash
            print(f"Crashed into at t={t}s")
            break



    # Plot
    fig, ax = plt.subplots(figsize=(10, 10)) #Creates first figure and axes for the trajectory graph
    ax.set_aspect('equal')  #Makes the x and y scale values equal
    ax.plot(craft_pos[:, 0], craft_pos[:, 1], 'r-', lw=1, label='Craft trajectory') #Plots spacecraft's path
    ax.plot(moon_pos[:, 0], moon_pos[:, 1], 'gray', lw=1, ls='--', label='Moon trajectory')  #Plots Moon's path

    # Earth and final Moon position
    ax.add_patch(plt.Circle((0, 0), rEarth, color='blue', label='Earth')) #Creates Earth on the graph, at (0,0) as a blue dot
    ax.add_patch(plt.Circle(moon_pos[-1], rMoon, color='gray', label='Moon (final)')) #Creates the final Moon position
    ax.add_patch(plt.Circle(moon_pos[tend*dt],rMoon,color='red',label='Moon @ timeburn1')) #Creates Moon at burn 1
    ax.add_patch(plt.Circle(moon_pos[tend2*dt],rMoon,color='orange',label='Moon @ timeburn2')) #Creates Moon at burn 2
    

    # Mark start
    ax.plot(craft_pos[0, 0], craft_pos[0, 1], 'g^', markersize=10, label='Start')  #Starting Point
    ax.plot(craft_pos[-1, 0], craft_pos[-1, 1], 'rv', markersize=10, label='End')  #End point

    def pos_at_time(t_target): #finds the position of the crat at a given time
        idx = int(t_target / dt) #finds time
        if idx >= len(craft_pos): #returns none if the time exceeds list legnth
            return None, None
        return craft_pos[idx], idx #returns the position of craft, at time given

    burns = [ #list of burn info
        (tstart,  'Burn 1 start',  'red',    +1),
        (tend,    'Burn 1 end',    'red',    +1),
        (tstart2, 'Burn 2 start',  'orange', -1),
        (tend2,   'Burn 2 end',    'orange', -1),
    ]

    for t_burn, label, color, sign in burns: 
        p, idx = pos_at_time(t_burn) #positon of craft, index time
        if p is None: #if there is no return, just continue
            continue
        # marker at burn location
        ax.plot(p[0], p[1], 'o', color=color, markersize=8, zorder=5)
        # label with timestamp
        ax.annotate(f'{label}\nt={t_burn}s',
                    xy=(p[0], p[1]),
                    xytext=(15, 15), textcoords='offset points', #offsets label
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8))
        #  thrust direction arrow
        v = craft_vel[idx] #finds velocity at the index
        vhat = v / np.linalg.norm(v) # finds direction
        arrow_len = 1.5e5 #scales arrow
        ax.arrow(p[0], p[1], sign*vhat[0]*arrow_len, sign*vhat[1]*arrow_len,
                head_width=4e4, color=color, alpha=0.6, zorder=4) #displays arrow

    ax.set_xlabel('x (m)') #Labels x-axis
    ax.set_ylabel('y (m)')  #Labels y-axisa
    ax.set_title(f'Earth-Moon Trajectory ({t_total/3600:.2f}hrs)') #Adds the plot title
    ax.legend()  #Creates a legend
    ax.grid(alpha=0.3)  #Adds grid

    fig, ax = plt.subplots(figsize=(10, 5))  #Creates a second plot
    ax.plot(times, craft_mass, 'b-', lw=1.5) #PLots spacecraft mass over time

    # Mark burn windows with shaded regions
    ax.axvspan(tstart, tend, alpha=0.3, color='red', label='Burn 1 (prograde)') #burn 1 labeled red
    ax.axvspan(tstart2, tend2, alpha=0.3, color='orange', label='Burn 2 (retrograde)') #burn 2 labeld orange

    # Reference line for dry mass
    ax.axhline(massPayload, color='gray', ls='--', lw=1, label=f'Dry mass ({massPayload:.0f} kg)') #dotted line to show payload dry mass

    ax.set_xlabel('Time (s)')  #Labels x-axis
    ax.set_ylabel('Craft mass (kg)') #Labels y-axis
    ax.set_title('Spacecraft Mass vs Time')#Adds the plot title
    ax.legend()#Creates a legend
    ax.grid(alpha=0.3) #Creates a grid

    idx_closest = np.argmin(dscm) #returns the index of the closest distance
    t__closest = idx_closest*dt # returns time of closest altitude
    print(f"Closest lunar approach {round((np.min(dscm-rMoon)/1e3),2)} km") #print
    print(f"At time:{t__closest}") # print
    plt.show() # shows our graphs


