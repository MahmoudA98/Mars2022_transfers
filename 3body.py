"""
===========================
Three-body example
===========================
"""
from numpy import sin, cos
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

n=3

# create a time array up to t_max sampled at steps of dt
dt=0.01
t_min=0.
t_max=50.
nstep=np.int((t_max-t_min)/dt)
print(n,nstep)
xp=np.zeros([n,nstep+1])
yp=np.zeros([n,nstep+1])

G=1.  # scaling of the potential
m=np.ones(n)  # mass of particles
eps=0.1  # gravitational softening
iseed=21
np.random.seed(iseed)

# set the initial 2D Cartesian coordinates and speeds
#x=np.zeros(n)
#y=np.zeros(n)
#vx=np.zeros(n)
#vy=np.zeros(n)
x=np.random.rand(1,n)[0]
y=np.random.rand(1,n)[0]
vx=np.random.rand(1,n)[0]
vy=np.random.rand(1,n)[0]

# shift to the centre of mass frame
xbar=np.sum(m*x)/np.sum(m)
x=x-xbar
ybar=np.sum(m*y)/np.sum(m)
y=y-ybar

# shift to the centre of momentum frame
vxbar=np.sum(m*vx)/np.sum(m)
vx=vx-vxbar
vybar=np.sum(m*vy)/np.sum(m)
vy=vy-vybar
vxbar=np.sum(m*vx)/np.sum(m)
vx=vx-vxbar
vybar=np.sum(m*vy)/np.sum(m)
vy=vy-vybar

def sep(x1,x2,y1,y2,eps):  # function for 2D separation
    return np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+eps*eps)

# determine energies
s01=sep(x[0],x[1],y[0],y[1],eps)
s12=sep(x[1],x[2],y[1],y[2],eps)
s20=sep(x[2],x[0],y[2],y[0],eps)
pe=-G*(m[0]*(m[1]/s01+m[2]/s20)+m[1]*m[2]/s12)
ke=np.sum(0.5*m*(vx*vx+vy*vy))
virial=-2*ke/pe
# rescale speeds so they are in keeping with the virial theorem
vx=vx/np.sqrt(virial)
vy=vy/np.sqrt(virial)
ke=np.sum(0.5*m*(vx*vx+vy*vy))
print(ke,pe,2*ke/pe)

print(x)
print(y)
print(vx)
print(vy)

step=0
while step<nstep:  # integrate forward in time
    fx=np.zeros(n)
    fy=np.zeros(n)
    for i in range(0,n-1):  # loop over all but the final particles
        for j in range(i+1,n):  # only the later particles in array
            s=sep(x[i],x[j],y[i],y[j],eps)
            fxij=-G*m[i]*m[j]*(x[i]-x[j])/(s*s*s)  # calculate forces
            fyij=-G*m[i]*m[j]*(y[i]-y[j])/(s*s*s)
            fx[i]=fx[i]+fxij
            fy[i]=fy[i]+fyij
            fx[j]=fx[j]-fxij   # use N3L to save calculations
            fy[j]=fy[j]-fyij
    vx=vx+fx*dt/m   # update velocities and positions
    vy=vy+fy*dt/m
    x=x+vx*dt
    y=y+vy*dt
#    s01=sep(x[0],x[1],y[0],y[1],eps)
#    s12=sep(x[1],x[2],y[1],y[2],eps)
#    s20=sep(x[2],x[0],y[2],y[0],eps)
#    pe=-G*(m[0]*(m[1]/s01+m[2]/s20)+m[1]*m[2]/s12)
#    ke=np.sum(0.5*m*(vx*vx+vy*vy))
#    print((step+1)*dt,pe+ke)  # check energy conservation
    step=step+1
    xp[:,step]=x  # store output positions
    yp[:,step]=y

int_ms=10 # interval between frames in ms
lag=50

fig=plt.figure(figsize=(12,12))

xmin=-1.5
xmax=-xmin
ymin=xmin
ymax=xmax
ax=fig.add_subplot(111,autoscale_on=False,xlim=(xmin,xmax),ylim=(ymin,ymax))
ax.set_aspect('equal')
orbit0,=ax.plot([],[],'o-',lw=2)
trail0,=ax.plot([],[],'b-',lw=1)
orbit1,=ax.plot([],[],'o-',lw=2)
trail1,=ax.plot([],[],'r-',lw=1)
orbit2,=ax.plot([],[],'o-',lw=2)
trail2,=ax.plot([],[],'g-',lw=1)
x0,y0,x1,y1,x2,y2=[],[],[],[],[],[]

def animate_all(i):
    modified=[] # List of objects in the plot that we've modified
    if i>lag:
        imin=i-lag
    else:
        imin=0
    thisx=[xp[0,i]]
    thisy=[yp[0,i]]
    orbit0.set_data(thisx,thisy)
    x0=xp[0,imin:i]
    y0=yp[0,imin:i]
    trail0.set_data(x0,y0)
    thisx=[xp[1,i]]
    thisy=[yp[1,i]]
    orbit1.set_data(thisx,thisy)
    x1=xp[1,imin:i]
    y1=yp[1,imin:i]
    trail1.set_data(x1,y1)
    thisx=[xp[2,i]]
    thisy=[yp[2,i]]
    orbit2.set_data(thisx,thisy)
    x2=xp[2,imin:i]
    y2=yp[2,imin:i]
    trail2.set_data(x2,y2)
    modified+=[orbit0,trail0,orbit1,trail1,orbit2,trail2]
    return modified

ani=animation.FuncAnimation(fig,animate_all,range(1,nstep),interval=int_ms,blit=True,repeat=False)

plt.show()
