import numpy as np
import matplotlib.pyplot as plt

G = 6.67e-11
Msun= 1.989e30
AU = 1.496e11
DAY = 24*60*60

w0 = -np.cbrt(2)/(2-np.cbrt(2))
w1 = 1/(2-np.cbrt(2))

c1 = w1/2
c4 = w1/2
c2 = (w0+w1)/2
c3 = (w0+w1)/2
d1 = w1
d3 = w1
d2 = w0

class Planet:                       #Keep Initial Conditions organized

    def __init__(self,name,e,sma,T,mass):

        self.name = name
        self.e = e                  #Eccentricity of orbit
        self.sma = sma              #Semimajor Axis (m)
        self.T = T                  #Orbital Period (s)\
        self.mass = mass
        self.history = [self.initial_conditions()]

    def initial_conditions(self):
        return np.array([self.sma*(1-self.e),0,0,-np.sqrt((G*Msun)/self.sma * ((1+self.e)/(1-self.e)))])

    def rhs(self, vector, planet_positions):
     x, y, u, v = vector

     r = np.sqrt(x ** 2 + y ** 2)

     dxdt = u
     dydt = v

     dudt = (-G * Msun * x) / r ** 3
     dvdt = (-G * Msun * y) / r ** 3

     for planet, position in planet_positions.items():
        if planet != self:
            dx = x - position[0]
            dy = y - position[1]
            dr = np.sqrt(dx ** 2 + dy ** 2)
            dudt -= (G * planet.mass * dx) / dr ** 3
            dvdt -= (G * planet.mass * dy) / dr ** 3

     return np.array([dxdt, dydt, dudt, dvdt])

    def yoshida(self, tmax, dt, planets):
        t = 0.0
        history = [self.initial_conditions()]
        times = [t]

        while t < tmax:

         if t + dt > tmax:
            dt = tmax - t

         state_old = history[-1]

         planet_positions = {planet: planet.history[-1][:2] for planet in planets}

         dvec = self.rhs(state_old, planet_positions)

         x1 = state_old[0] + c1 * dvec[0] * dt
         y1 = state_old[1] + c1 * dvec[1] * dt
         vx1 = state_old[2] + d1 * dvec[2] * dt
         vy1 = state_old[3] + d1 * dvec[3] * dt
 
         mid_state = np.array([x1, y1, vx1, vy1])
         dvec_mid = self.rhs(mid_state, planet_positions)

         x2 = x1 + c2 * dvec_mid[0] * dt
         y2 = y1 + c2 * dvec_mid[1] * dt
         vx2 = vx1 + d2 * dvec_mid[2] * dt
         vy2 = vy1 + d2 * dvec_mid[3] * dt

         next_state = np.array([x2, y2, vx2, vy2])
         dvec_next = self.rhs(next_state, planet_positions)

         x3 = x2 + c3 * dvec_next[0] * dt
         y3 = y2 + c3 * dvec_next[1] * dt
         vx3 = vx2 + d3 * dvec_next[2] * dt
         vy3 = vy2 + d3 * dvec_next[3] * dt

         end_state = np.array([x3, y3, vx3, vy3])
         dvec_end = self.rhs(end_state, planet_positions)

         xp1 = x3 + c4 * dvec_end[0] * dt
         yp1 = y3 + c4 * dvec_end[1] * dt
         vxp1 = vx3
         vyp1 = vy3

         state_new = np.array([xp1, yp1, vxp1, vyp1])

         t += dt

         times.append(t)
         history.append(state_new)

        return times, history
    

    def coords(self,dt,planets):

         self.times, self.history = self.yoshida(self.T,dt,planets)

         self.x_pos = [q[0]/AU for q in self.history]
         self.y_pos = [q[1]/AU for q in self.history]
     
         return self.x_pos, self.y_pos
    
    def error(self):
    
     R_orig = np.sqrt(self.x_pos[0]**2 + self.y_pos[0]**2)
     R_new = np.sqrt(self.x_pos[-1]**2 + self.y_pos[-1]**2)
     e = np.abs(R_new - R_orig)
    
     return e


venus = Planet("Venus",0.007,0.723*AU,230*DAY,4.86732e24)
earth = Planet("Earth",0.0167,AU,368*DAY,5.972e24)
mars = Planet("Mars",0.093,1.534*AU,696*DAY,6.39e23)
jupiter = Planet("Jupiter",0.049,5.2*AU,4333*DAY,1.898e27)
saturn = Planet("Saturn",0.0565,9.5826*AU,10770*DAY,5.68e26)

ax = plt.subplot(111)
ax.set_title(r"Yoshida Integration Test for Solar System Orbits")
ax.set_xlabel(r"x [AU]")
ax.set_ylabel(r"y [AU]")
ax.scatter([0], [0], marker=(10,1), color="y", s=250)

Planets = [venus,earth,mars,jupiter,saturn]

for p in Planets:

  if (p.T < 700):
     x_pos, y_pos = p.coords(100, Planets)  # Change time step for more accuracy at the expense of longer simulation time
     ax.plot(x_pos, y_pos, label=p.name) 
     print(f'Planet = {p.name}, Error = {p.error()}')

  else:
     x_pos, y_pos = p.coords(1000, Planets)  # Change time step for more accuracy at the expense of longer simulation time
     ax.plot(x_pos, y_pos, label=p.name) 
     print(f'Planet = {p.name}, Error = {p.error()}')

ax.legend()
plt.show()

time_steps = [500, 700, 1000, 2000, 5000, 10000]

errors = {planet.name: [] for planet in Planets}

for dt in time_steps:
    for p in Planets:
        x_pos, y_pos = p.coords(dt, Planets)
        error = p.error()
        errors[p.name].append(error)

fig, bx = plt.subplots()
bx.set_title(r"Error vs Time Step")
bx.set_xlabel(r"Time Step (s)")
bx.set_ylabel(r"Error (AU)")

for planet, error_list in errors.items():
    bx.plot(time_steps, error_list, label=planet)

bx.legend()
plt.show()
