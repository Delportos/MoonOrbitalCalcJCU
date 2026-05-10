import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
tstart = 3119
tend = tstart + t1delta

t2delta = 130
tstart2 = 151940
tend2 = tstart2 + t2delta
tstart3, tend3 = 28700, 28720

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

# History arrays
craft_pos = np.zeros((n_steps, 2))
moon_pos  = np.zeros((n_steps, 2))
craft_mass = np.zeros(n_steps)
craft_vel = np.zeros((n_steps, 2))
times = np.arange(n_steps) * dt
dscm = np.zeros(n_steps)
burning = np.zeros(n_steps, dtype=bool)   # for animation: was the engine firing this step?

crashed = False
crash_step = None


def CraftAccel(posCraft, posEarth, posMoon):
    r_e = posCraft - posEarth
    r_e_mag = np.linalg.norm(r_e)
    a_earth = -G * mEarth * r_e / r_e_mag**3
    r_m = posCraft - posMoon
    r_m_mag = np.linalg.norm(r_m)
    a_moon = -G * mMoon * r_m / r_m_mag**3
    return a_earth + a_moon


def moonAccel(posMoon, posEarth):
    r = posMoon - posEarth
    rmag = np.linalg.norm(r)
    return -G * mEarth * r / rmag**3


# ============ SIMULATION ============
for i in range(n_steps):
    t = i * dt

    a_craft = CraftAccel(posCraft, posEarth, posMoon)
    a_moon  = moonAccel(posMoon, posEarth)

    vCraftmid = vCraft + 0.5 * a_craft * dt
    vMoonmid  = vMoon  + 0.5 * a_moon  * dt

    posCraft = posCraft + vCraftmid * dt
    posMoon  = posMoon  + vMoonmid  * dt

    a_craft = CraftAccel(posCraft, posEarth, posMoon)
    a_moon  = moonAccel(posMoon, posEarth)

    vCraft = vCraftmid + 0.5 * a_craft * dt
    vMoon  = vMoonmid  + 0.5 * a_moon  * dt

    is_burning = False
    if tstart < t < tend:
        vhat = vCraft / np.linalg.norm(vCraft)
        vCraft = vCraft + vhat * thrustmag / mCraft * dt
        mCraft -= mdot * dt
        is_burning = True
    if tstart2 < t < tend2:
        vhat = vCraft / np.linalg.norm(vCraft)
        vCraft = vCraft - vhat * thrustmag / mCraft * dt
        mCraft -= mdot * dt
        is_burning = True

    craft_pos[i] = posCraft
    moon_pos[i]  = posMoon
    craft_mass[i] = mCraft
    craft_vel[i] = vCraft
    dscm[i] = np.linalg.norm(posCraft - posMoon)
    burning[i] = is_burning

    # Earth crash
    if np.linalg.norm(posCraft) < rEarth + 1e5:
        print(f"Crashed into Earth at t={t}s")
        crashed = True
        crash_step = i
        break
    # Moon crash
    if np.linalg.norm(posCraft - posMoon) < rMoon:
        print(f"Crashed into Moon at t={t}s")
        crashed = True
        crash_step = i
        break

# Trim if crashed
if crashed:
    craft_pos  = craft_pos[:crash_step+1]
    moon_pos   = moon_pos[:crash_step+1]
    craft_mass = craft_mass[:crash_step+1]
    craft_vel  = craft_vel[:crash_step+1]
    dscm       = dscm[:crash_step+1]
    times      = times[:crash_step+1]
    burning    = burning[:crash_step+1]

n_frames_total = len(craft_pos)
print(f"Simulation complete. {n_frames_total} steps recorded.")
print(f"Closest lunar approach: {dscm.min()/1000:.1f} km (center-to-center)")
print(f"  Altitude above surface: {(dscm.min() - rMoon)/1000:.1f} km")
print(f"  At t = {np.argmin(dscm) * dt}s")


# ============ ANIMATION (Craft-centered frame) ============
frame_skip = 60
frame_indices = np.arange(0, n_frames_total, frame_skip)

# Transform all positions into craft-centered frame
moon_pos_cc  = moon_pos - craft_pos     # moon relative to craft
earth_pos_cc = -craft_pos               # earth relative to craft (posEarth = origin)

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
# View extent: a bit more than Earth-Moon distance, so both fit when craft is in LEO
ax.set_xlim(-5e8, 5e8)
ax.set_ylim(-5e8, 5e8)
ax.set_xlabel('x (m, craft-centered)')
ax.set_ylabel('y (m, craft-centered)')
ax.grid(alpha=0.3)

# Craft is now stationary at origin
craft_dot, = ax.plot([0], [0], 'ro', markersize=6, zorder=5, label='Craft (fixed)')

# Moon and Earth are now the moving bodies
moon_circ  = plt.Circle(moon_pos_cc[0],  rMoon,  color='gray', zorder=3)
earth_circ = plt.Circle(earth_pos_cc[0], rEarth, color='blue', zorder=3)
ax.add_patch(moon_circ)
ax.add_patch(earth_circ)

# Trails of the OTHER bodies as seen from craft
moon_trail,  = ax.plot([], [], 'gray', ls='--', lw=0.6, alpha=0.5, label='Moon path')
earth_trail, = ax.plot([], [], 'b-', lw=0.5, alpha=0.4, label='Earth path')

# Flame stays at origin (where craft is)
flame, = ax.plot([], [], 'o', color='yellow', markersize=14, alpha=0.8,
                 markeredgecolor='orange', zorder=4)

time_text = ax.text(0.02, 0.97, '', transform=ax.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.legend(loc='upper right')
ax.set_title('Trajectory (Craft-centered frame)')


def init():
    moon_trail.set_data([], [])
    earth_trail.set_data([], [])
    flame.set_data([], [])
    time_text.set_text('')
    return craft_dot, moon_trail, earth_trail, moon_circ, earth_circ, flame, time_text


def animate(frame_idx):
    i = frame_indices[frame_idx]

    trail_start = max(0, i - 10000)

    # Moon and Earth positions relative to craft
    moon_circ.center  = (moon_pos_cc[i, 0],  moon_pos_cc[i, 1])
    earth_circ.center = (earth_pos_cc[i, 0], earth_pos_cc[i, 1])

    moon_trail.set_data(moon_pos_cc[trail_start:i+1, 0],
                        moon_pos_cc[trail_start:i+1, 1])
    earth_trail.set_data(earth_pos_cc[trail_start:i+1, 0],
                         earth_pos_cc[trail_start:i+1, 1])

    # Flame at origin during burns
    if burning[i]:
        flame.set_data([0], [0])
    else:
        flame.set_data([], [])

    t_now = i * dt
    hrs = t_now // 3600
    mns = (t_now % 3600) // 60
    surf_alt = (dscm[i] - rMoon) / 1000
    time_text.set_text(
        f't = {t_now:>6.0f}s  ({hrs:.0f}h {mns:.0f}m)\n'
        f'Mass: {craft_mass[i]:>8.0f} kg\n'
        f'Moon dist: {dscm[i]/1000:>7.0f} km\n'
        f'Surface alt: {surf_alt:>7.0f} km'
    )
    return craft_dot, moon_trail, earth_trail, moon_circ, earth_circ, flame, time_text


ani = animation.FuncAnimation(
    fig, animate, init_func=init,
    frames=len(frame_indices),
    interval=20, blit=False, repeat=True
)

plt.show()

ani.save('finalcodes/vids/satstationarytrajectory.mp4', fps=30, dpi=120)