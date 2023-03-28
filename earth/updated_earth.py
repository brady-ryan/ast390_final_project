import numpy as np
import matplotlib.pyplot as plt

G = 6.67e-11
Msun= 1.989e30
AU = 1.496e11

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

    def __init__(self,e,sma,T):

        self.e = e                  #Eccentricity of orbit
        self.sma = sma              #Semimajor Axis (m)
        self.T = T                  #Orbital Period (s)

    def vec(self):
        return np.array([self.sma*(1-self.e),0,0,-np.sqrt((G*Msun)/self.sma * ((1+self.e)/(1-self.e)))])

    def rhs(self,vector):
        self.x,self.y,self.u,self.v = vector

        r = np.sqrt(self.x**2+self.y**2)

        dxdt = self.u
        dydt = self.v

        dudt = (-G*Msun*self.x)/r**3
        dvdt = (-G*Msun*self.y)/r**3

        return np.array([dxdt,dydt,dudt,dvdt])

    def yoshida(self,tmax,dt):

     t = 0.0
     history = [self.vec()]
     times = [t]

     while t < tmax:
         if t + dt > tmax:
             dt = tmax - t

         state_old = history[-1]

         dvec = self.rhs(state_old)

         x1 = state_old[0] + c1*dvec[0]*dt
         y1 = state_old[1] + c1*dvec[1]*dt
         vx1 = state_old[2] + d1*dvec[2]*dt
         vy1 = state_old[3] + d1*dvec[3]*dt

         mid_state = np.array([x1,y1,vx1,vy1])
         dvec_mid = self.rhs(mid_state)

         x2 = x1 + c2*dvec_mid[0]*dt
         y2 = y1 + c2*dvec_mid[1]*dt
         vx2 = vx1 + d2*dvec_mid[2]*dt
         vy2 = vy1 + d2*dvec_mid[3]*dt

         next_state = np.array([x2,y2,vx2,vy2])
         dvec_next = self.rhs(next_state)

         x3 = x2 + c3*dvec_next[0]*dt
         y3 = y2 + c3*dvec_next[1]*dt
         vx3 = vx2 + d3*dvec_next[2]*dt
         vy3 = vy2 + d3*dvec_next[3]*dt

         end_state = np.array([x3,y3,vx3,vy3])
         dvec_end = self.rhs(end_state)

         xp1 = x3 + c4*dvec_end[0]*dt
         yp1 = y3 + c4*dvec_end[1]*dt
         vxp1 = vx3
         vyp1 = vy3

         state_new = np.array([xp1,yp1,vxp1,vyp1])

         t += dt

         times.append(t)
         history.append(state_new)

     return times, history

earth = Planet(0.0167,AU,2*365*24*60*60)
etimes, ehistory = Planet.yoshida(earth,earth.T,100)

x_pos = [v[0]/AU for v in ehistory]
y_pos = [v[1]/AU for v in ehistory]

ax = plt.subplot(111)
ax.set_title(r"Yoshida Integration Test for Earth's Orbit")
ax.set_xlabel(r"x [AU]")
ax.set_ylabel(r"y [AU]")
ax.plot(x_pos,y_pos)
ax.scatter([0], [0], marker=(10,1), color="y", s=250)

plt.show()
