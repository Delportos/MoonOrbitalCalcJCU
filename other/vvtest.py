import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.674e-11
mEarth = 5.972e24
mMoon = 7.35e22
rEarth = 6.4e6
muE = 3.98600436e14
muM = 4.903e12
d = 384.4e6  # Earth-Moon distance

# Craft
h = 4e5
massPayload = 1e4
massFuel = 26298.85
wetMass = massPayload + massFuel
isp = 340.
exhaustv = 9.81 * isp

# Burn timing — note: 10 s is very short for a TLI; you may want ~300-600 s
tstart = 4070.
burntimediff = 100.
tend = tstart + burntimediff
mdot = massFuel / burntimediff   # was dividing by burntime=60000, inconsistent with burntimediff

# Moon orbit (analytical, circular)
omegaMoon = np.sqrt(G * mEarth / d**3)
thetaM0 = - np.pi / 2 * .2  # Moon starts on -x axis

def moon_pos(t):
    return d * np.array([np.cos(omegaMoon * t + thetaM0),
                         np.sin(omegaMoon * t + thetaM0)])

# Initial craft state — match the Moon's starting side
posCraft0 = np.array([rEarth + h, 0.0])
vCraft0   = np.array([0.0, 7.66e3])

# Time grid
tf = 6 * 24 * 3600.  # 6 days, plenty for a translunar coast
dt = 20.
t = np.arange(0, tf, dt)
N = len(t)

def acceleration(x, t, mass, thrust_dir=None, thrusting=False):
    r_em = x
    r_mc = x - moon_pos(t)
    a = (-muE * r_em / np.linalg.norm(r_em)**3
         - muM * r_mc / np.linalg.norm(r_mc)**3)
    if thrusting and thrust_dir is not None:
        thrust_force = mdot * exhaustv
        a = a + (thrust_force / mass) * thrust_dir
    return a

def velocity_verlet(t, x0, v0):
    x = np.zeros((len(t), 2))
    v = np.zeros((len(t), 2))
    x[0], v[0] = x0, v0
    mass = wetMass
    dt = t[1] - t[0]
    for i in range(len(t) - 1):
        thrusting = tstart <= t[i] <= tend
        thrust_dir = v[i] / np.linalg.norm(v[i]) if thrusting else None
        a_i = acceleration(x[i], t[i], mass, thrust_dir, thrusting)

        v_half = v[i] + 0.5 * dt * a_i
        x[i+1] = x[i] + dt * v_half

        if thrusting:
            mass = max(mass - mdot * dt, massPayload)

        thrusting_next = tstart <= t[i+1] <= tend
        thrust_dir_next = v_half / np.linalg.norm(v_half) if thrusting_next else None
        a_next = acceleration(x[i+1], t[i+1], mass, thrust_dir_next, thrusting_next)
        v[i+1] = v_half + 0.5 * dt * a_next
    return x, v

xVV, vVV = velocity_verlet(t, posCraft0, vCraft0)
moon_track = np.array([moon_pos(ti) for ti in t])

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(0, 0, 'bo', markersize=10, label="Earth")
ax.plot(moon_track[:, 0], moon_track[:, 1], 'k--', alpha=0.4, label="Moon's path")
ax.plot(moon_track[-1, 0], moon_track[-1, 1], 'ko', markersize=6, label="Moon (final)")
ax.plot(xVV[:, 0], xVV[:, 1], 'r-', lw=0.8, label="Craft trajectory")
ax.set_aspect("equal")
ax.set_xlabel("X position (m)")
ax.set_ylabel("Y position (m)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()