#Ryan Brady
#AST 390 Final Project
#Simulating the Exoplanetary Motions around Tau Ceti

import numpy as np
import matplotlib.pyplot as plt

#ALL UNITS IN MKS

G = 6.67e-11        #Newton's Gravitational Constat
Msun = 1.989e30     #Solar Mass
Mearth = 5.972e24   #Earth Mass
M = 0.783 * Msun    #Tau Ceti Mass
AU = 1.496e11       #Astronomical Unit
DAY = 24*60*60      #1 Day in seconds

w0 = -np.cbrt(2)/(2-np.cbrt(2))   #Yoshida w0 constant
w1 = 1/(2-np.cbrt(2))             #Yoshida w1 constant

c1 = w1/2        #Yoshida c1 constant
c4 = w1/2        #Yoshida c4 constant
c2 = (w0+w1)/2   #Yoshida c2 constant
c3 = (w0+w1)/2   #Yoshida c3 constant
d1 = w1          #Yoshida d1 costant
d3 = w1          #Yoshida d3 constant
d2 = w0          #Yoshida d2 constant

class Planet:          
    '''Class to organize initial conditions, 
    conduct the Yoshida integration, extract
    the final positions, and calculate error.'''          

    def __init__(self,name,e,sma,T,mass):
        '''User input initial conditions'''

        self.name = name      #Planet name
        self.e = e            #Planet eccentricity    
        self.sma = sma        #Planet semi-major axis          
        self.T = T            #Planet orbital period       
        self.mass = mass      #Planet mass
        self.history = [self.initial_conditions()]  

    def initial_conditions(self):
        '''Calculate initial x,y positions and v_x,v_y velocities'''
        return np.array([self.sma*(1-self.e),0,0,
                         -np.sqrt((G*M)/self.sma * ((1+self.e)/(1-self.e)))])

    def rhs(self, vector, planet_positions):
     '''Calculate change in x,y positions and 
     change in v_x, v_y velocities.'''
     x, y, u, v = vector

     r = np.sqrt(x**2 + y**2)

     dxdt = u
     dydt = v

     dudt = (-G * M * x) / r**3
     dvdt = (-G * M * y) / r**3

     for planet, position in planet_positions.items():
        if planet != self:
            '''Calculate effects of all bodies in system'''
            dx = x - position[0]
            dy = y - position[1]
            dr = np.sqrt(dx**2 + dy**2)
            dudt -= (G * planet.mass * dx) / dr**3
            dvdt -= (G * planet.mass * dy) / dr**3

     return np.array([dxdt, dydt, dudt, dvdt])

    def yoshida(self, tmax, dt, planets):
        '''Main Yoshida integrator'''
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
         '''Extract x,y positions from Yoshida integration
         by taking the corresponding array elements of "history"'''

         self.times, self.history = self.yoshida(self.T,dt,planets)

         self.x_pos = [q[0]/AU for q in self.history]
         self.y_pos = [q[1]/AU for q in self.history]
     
         return self.x_pos, self.y_pos
    
    def error(self):
     '''Calculate error by subtracting initial 
     and final positions of simulation'''
    
     R_orig = np.sqrt(self.x_pos[0]**2 + self.y_pos[0]**2)
     R_new = np.sqrt(self.x_pos[-1]**2 + self.y_pos[-1]**2)
     e = np.abs(R_new - R_orig)
    
     return e


#Initialize all planets with initial conditions
b = Planet("b",0.16,0.105*AU,13.965*DAY,2*Mearth)
g = Planet('g',0.06,0.133*AU,20*DAY,1.75*Mearth)
c = Planet('c',0.03,0.195*AU,35.362*DAY,3.1*Mearth)
h = Planet('h',0.23,0.243*AU,49.41*DAY,1.83*Mearth)
d = Planet('d',0.08,0.374*AU,94.11*DAY,3.6*Mearth)
e = Planet('e',0.18,0.538*AU,162.87*DAY,3.93*Mearth)
f = Planet('f',0.16,1.334*AU,636*DAY,3.93*Mearth)

#Construct plot
ax = plt.subplot(111)
ax.set_title(r"Exoplanetary Motions Around $\tau$ Ceti")
ax.set_xlabel(r"x [AU]")
ax.set_ylabel(r"y [AU]")
ax.scatter([0], [0], marker=(10,1), color="y", s=250)


Planets = [b,g,c,h,d,e,f]   #Organize planets to one list


for p in Planets:

    x_pos, y_pos = p.coords(1000,Planets)   #Calculate simulation coordinates for each planet

    if p.name == "b" or p.name == "c" or p.name == "d":
        ax.plot(x_pos,y_pos,":",label = f"{p.name} (Theoretical)")
    else:
       ax.plot(x_pos,y_pos,label=p.name)

    print(f'Planet = {p.name}, Error = {p.error()}')

ax.legend()
plt.show()


time_steps = [100, 500, 1000, 2000, 5000, 10000]

errors = {planet.name: [] for planet in Planets}

#Calculate errors at multiple timesteps
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