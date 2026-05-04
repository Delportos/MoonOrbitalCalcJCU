import numpy as np
import matplotlib.pyplot as plt

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

# Burn windows
t1delta = 520
tstart =  3119
tend = tstart+t1delta

t2delta= 130
tstart2 = 151940 
tend2 = tstart2 +t2delta
tstart3,tend3 = 28700,28720

# Sim params
dt = 1
t_total = 2.4e5
n_steps = int(t_total / dt)

# Initial state
posCraft = np.array([rEarth + h, 0.0])
vCraft   = np.array([0.0, 7.66e3])
posMoon  = np.array([d, 0.0])
vMoon    = np.array([0.0, 1.022e3])
mCraft   = wetMass
posEarth = np.array([0.0, 0.0])

# Preallocated history arrays
craft_pos = np.zeros((n_steps, 2))
moon_pos  = np.zeros((n_steps, 2))
craft_mass = np.zeros(n_steps)
craft_vel = np.zeros((n_steps, 2))
times = np.arange(n_steps) * dt
dscm = np.zeros((n_steps,1))

crashed = False
crash_step = None

def CraftAccel(posCraft, posEarth,posMoon):
    # Gravity from Earth on craft
    r_e = posCraft - posEarth
    r_e_mag = np.linalg.norm(r_e)
    a_earth = -G * mEarth * r_e / r_e_mag**3   # combines rhat and 1/r^2

    # Gravity from Moon on craft
    r_m = posCraft - posMoon
    r_m_mag = np.linalg.norm(r_m)
    a_moon = -G * mMoon * r_m / r_m_mag**3

    acceleration = a_earth + a_moon #acc1
    return acceleration

def moonAccel(posMoon,posEarth):
    r = posMoon - posEarth
    rmag = np.linalg.norm(r)
    a = -G*mEarth*r / rmag**3

    return a

for i in range(n_steps):
    t = i * dt

    a_craft = CraftAccel(posCraft,posEarth,posMoon)# Gravity from Earth on craft
    a_moon = moonAccel(posMoon,posEarth)

    vCraftmid = vCraft + 0.5*a_craft*dt
    vMoonmid = vMoon + 0.5*a_moon*dt

    posCraft = posCraft + vCraftmid * dt
    posMoon = posMoon + vMoonmid * dt


    a_craft = CraftAccel(posCraft,posEarth,posMoon)# Gravity from Earth on craft
    a_moon = moonAccel(posMoon,posEarth)

    vCraft = vCraftmid + 0.5*a_craft * dt #acceleration 2
    vMoon = vMoonmid + 0.5*a_moon * dt

    # Burn 1 (prograde)
    if tstart < t < tend:
        vhat = vCraft / np.linalg.norm(vCraft)
        vCraft = vCraft + vhat * thrustmag/mCraft * dt
        mCraft -= mdot * dt

    # Burn 2 (retrograde)
    if tstart2 < t < tend2:
        vhat = vCraft / np.linalg.norm(vCraft)
        vCraft = vCraft - vhat * thrustmag/mCraft * dt
        mCraft -= mdot * dt
    """
    #burn 3
    if tstart3 < t < tend3:
        vhat = vCraft / np.linalg.norm(vCraft)
        vCraft = vCraft - vhat * thrustmag/mCraft * dt
        mCraft -= mdot * dt
"""
    
    # Store
    craft_pos[i] = posCraft
    moon_pos[i]  = posMoon
    craft_mass[i] = mCraft
    craft_vel[i] = vCraft 
    dscm[i] = np.linalg.norm(posCraft - posMoon)

    # Crash check
    if np.linalg.norm(posCraft) < rEarth + 1e5:
        print(f"Crashed at t={t}s")
        crashed = True
        crash_step = i
        break

# Trim if crashed
if crashed:
    craft_pos = craft_pos[:crash_step+1]
    moon_pos  = moon_pos[:crash_step+1]
    craft_mass = craft_mass[:crash_step+1]
    times = times[:crash_step+1]

# Plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.plot(craft_pos[:, 0], craft_pos[:, 1], 'r-', lw=1, label='Craft trajectory')
ax.plot(moon_pos[:, 0], moon_pos[:, 1], 'gray', lw=1, ls='--', label='Moon trajectory')

# Earth and final Moon position
ax.add_patch(plt.Circle((0, 0), rEarth, color='blue', label='Earth'))
ax.add_patch(plt.Circle(moon_pos[-1], rMoon, color='gray', label='Moon (final)'))
ax.add_patch(plt.Circle(moon_pos[tend*dt],rMoon,color='red',label='Moon @ timeburn1'))
ax.add_patch(plt.Circle(moon_pos[tend2*dt],rMoon,color='orange',label='Moon @ timeburn2'))

# Mark start
ax.plot(craft_pos[0, 0], craft_pos[0, 1], 'g^', markersize=10, label='Start')
ax.plot(craft_pos[-1, 0], craft_pos[-1, 1], 'rv', markersize=10, label='End')

def pos_at_time(t_target):
    idx = int(t_target / dt)
    if idx >= len(craft_pos):
        return None, None
    return craft_pos[idx], idx

burns = [
    (tstart,  'Burn 1 start',  'red',    +1),
    (tend,    'Burn 1 end',    'red',    +1),
    (tstart2, 'Burn 2 start',  'orange', -1),
    (tend2,   'Burn 2 end',    'orange', -1),
    (tstart3, 'Burn 3 start',  'purple', -1),
    (tend3,   'Burn 3 end',    'purple', -1),
]

for t_burn, label, color, sign in burns:
    p, idx = pos_at_time(t_burn)
    if p is None:
        continue
    # Marker at burn location
    ax.plot(p[0], p[1], 'o', color=color, markersize=8, zorder=5)
    # Label with timestamp
    ax.annotate(f'{label}\nt={t_burn}s',
                xy=(p[0], p[1]),
                xytext=(15, 15), textcoords='offset points',
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=0.8))
    # Optional: thrust direction arrow
    v = craft_vel[idx]
    vhat = v / np.linalg.norm(v)
    arrow_len = 1.5e5
    ax.arrow(p[0], p[1], sign*vhat[0]*arrow_len, sign*vhat[1]*arrow_len,
             head_width=4e4, color=color, alpha=0.6, zorder=4)

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title(f'Earth-Moon Trajectory ({t_total}s)')
ax.legend()
ax.grid(alpha=0.3)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(times, craft_mass, 'b-', lw=1.5)

# Mark burn windows with shaded regions
ax.axvspan(tstart, tend, alpha=0.3, color='red', label='Burn 1 (prograde)')
ax.axvspan(tstart2, tend2, alpha=0.3, color='orange', label='Burn 2 (retrograde)')

# Reference line for dry mass
ax.axhline(massPayload, color='gray', ls='--', lw=1, label=f'Dry mass ({massPayload:.0f} kg)')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Craft mass (kg)')
ax.set_title('Spacecraft Mass vs Time')
ax.legend()
ax.grid(alpha=0.3)

idx_closest = np.argmin(dscm)
t__closest = idx_closest*dt
print(f"Closest lunar approach {round((np.min(dscm)/1e3),2)} km")
print(f"At time:{t__closest}")
plt.show()



