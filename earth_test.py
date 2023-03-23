import numpy as np
import matplotlib.pyplot as plt

G = 1
Msun = 37

w0 = -np.cbrt(2)/(2-np.cbrt(2))
w1 = 1/(2-np.cbrt(2))

c1 = w1/2
c4 = w1/2
c2 = (w0+w1)/2
c3 = (w0+w1)/2
d1 = w1
d3 = w1
d2 = w0


def rhs(vector):
    x,y,u,v = vector

    r = np.sqrt(x**2+y**2)

    dxdt = u
    dydt = v

    dudt = (-G*Msun*x)/r**3
    dvdt = (-G*Msun*y)/r**3

    return np.array([dxdt,dydt,dudt,dvdt])

def integrate(vec0,tmax,dt):

    t = 0.0
    history = [np.array(vec0)]
    times = [t]

    while t < tmax:
        if t + dt > tmax:
            dt = tmax - t

        state_old = history[-1]

        dvec = rhs(state_old)

        x1 = state_old[0] + c1*dvec[0]*dt
        y1 = state_old[1] + c1*dvec[1]*dt
        vx1 = state_old[2] + d1*dvec[2]*dt
        vy1 = state_old[3] + d1*dvec[3]*dt

        mid_state = np.array([x1,y1,vx1,vy1])
        dvec_mid = rhs(mid_state)

        x2 = x1 + c2*dvec_mid[0]*dt
        y2 = y1 + c2*dvec_mid[1]*dt
        vx2 = vx1 + d2*dvec_mid[2]*dt
        vy2 = vy1 + d2*dvec_mid[3]*dt

        next_state = np.array([x2,y2,vx2,vy2])
        dvec_next = rhs(next_state)

        x3 = x2 + c3*dvec_next[0]*dt
        y3 = y2 + c3*dvec_next[1]*dt
        vx3 = vx2 + d3*dvec_next[2]*dt
        vy3 = vy2 + d3*dvec_next[3]*dt

        end_state = np.array([x3,y3,vx3,vy3])
        dvec_end = rhs(end_state)

        xp1 = x3 + c4*dvec_end[0]*dt
        yp1 = y3 + c4*dvec_end[1]*dt
        vxp1 = vx3
        vyp1 = vy3

        state_new = np.array([xp1,yp1,vxp1,vyp1])

        t += dt

        times.append(t)
        history.append(state_new)

    return times, history


a = 1
e = .0167
earth = np.array([a*(1-e),0,0,-np.sqrt((G*Msun/a) * ((1+e)/(1-e)))])

t = 5
dt = 0.00001

etimes, ehistory = integrate(earth,t,dt)

x_pos = [v[0] for v in ehistory]
y_pos = [v[1] for v in ehistory]

ax = plt.subplot(111)
ax.plot(x_pos,y_pos)
ax.scatter([0], [0], marker=(10,1), color="y", s=250)

plt.show()