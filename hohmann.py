import numpy as np
import math
import matplotlib.pyplot as plt
G = 6.6726e-11   #Grav constant
dt=200  #timestep size
n= 111500  # *based on theorteical transfer time and dt=200
i=0 #for the loop later
PI=np.pi

class Planet:
    def __init__(self, name, mass, dist, radius, period,theta):
        self.name=name
        self.mass=mass
        self.dist=dist
        self.radius=radius
        self.period=period
        self.theta=theta  #
        self.pos=self.dist*np.array([np.cos(2*PI*dt*i/self.period+theta),np.sin(2*PI*dt*i/self.period+theta)])
        #angular offset based on 28/07/2022   https://www.theplanetstoday.com/

class Satellite:
    def __init__(self, name, mass, pos):
        self.name=name
        self.mass=mass
        self.pos=pos
        self.vel=np.array([0,0])
        
def toUnit(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def grav(pos, m1,m2,m3,m4): #m1 is sun, m2 earth, m3 mars, m4 venus
    acc=np.array([0.0,0.0])
    r1=m1.pos-pos
    r2=m2.pos-pos
    r3=m3.pos-pos
    r4=m4.pos-pos
    magr1=np.linalg.norm(r1)
    magr2=np.linalg.norm(r2)
    magr3=np.linalg.norm(r3)
    magr4=np.linalg.norm(r4)
    r11=toUnit(r1)
    r21=toUnit(r2)
    r31=toUnit(r3)
    r41=toUnit(r4)
    acc1_mag= G*m1.mass/(magr1*magr1)
    acc2_mag= G*m2.mass/(magr2*magr2)
    acc3_mag= G*m3.mass/(magr3*magr3)
    acc4_mag= G*m4.mass/(magr4*magr4)
    acc[0]= acc1_mag*r11[0]+acc2_mag*r21[0]+acc3_mag*r31[0]+acc4_mag*r41[0]
    acc[1]= acc1_mag*r11[1]+acc2_mag*r21[1]+acc3_mag*r31[1]+acc4_mag*r41[0]
    return acc

def hamiltonian(fly, m1,m2,m3):
    ham=(np.linalg.norm(fly.vel))**2
    r1=m1.pos-fly.pos
    r2=m2.pos-fly.pos
    r3=np.pos-fly.pos
    magr1=np.linalg.norm(r1)
    magr2=np.linalg.norm(r2)
    magr3=np.linalg.norm(r3)
    ham=ham-G*((m1/magr1)+(m2/magr2)+(m3/magr3))
    return ham

def RK4_nbody(fly,m1,m2,m3,m4,a):  #up to 4 bodies
    z1=fly.pos+(a/2)*fly.vel
    acc1=grav(fly.pos,m1,m2,m3,m4) #need fly but with position z1
    v1=fly.vel+(a/2)*acc1
    z2=fly.pos+(a/2)*v1
    acc2=grav(z1,m1,m2,m3,m4)   #need fly but with position z2
    v2=fly.vel+(a/2)*acc2
    z3=fly.pos+a*v2
    acc3=grav(z2,m1,m2,m3,m4)
    v3=fly.vel+a*acc3
    fly.pos=fly.pos+(a/6)*(fly.vel+2*v1+2*v2+v3)
    acc4=grav(z3,m1,m2,m3,m4)
    fly.vel= fly.vel+(a/6)*(acc1+2*acc2+2*acc3+acc4)
    
def boost(fly, v):
    fly.vel[0]+=v[0]
    fly.vel[1]+=v[1]

def KE_per_mass(fly):
    vv=np.linalg.norm(fly.vel)
    return 0.5*(vv**2)

def energy_correction(fly,m1,m2,m3,a):    #m1,m2,m3 are masses not accounted for in the vis-viva eqn and a is dt
    counter_acc=np.array([0.0,0.0])
    pos=fly.pos
    r1=m1.pos-pos
    r2=m2.pos-pos
    r3=m3.pos-pos
    magr1=np.linalg.norm(r1)
    magr2=np.linalg.norm(r2)
    magr3=np.linalg.norm(r3)
    acc1_mag= G*m1.mass/(magr1*magr1)
    acc2_mag= G*m2.mass/(magr2*magr2)
    acc3_mag= G*m3.mass/(magr3*magr3)
    counter_acc[0]= acc1_mag*r11[0]+acc2_mag*r21[0]+acc3_mag*r31[0]
    counter_acc[1]= acc1_mag*r11[1]+acc2_mag*r21[1]+acc3_mag*r31[1]
    distance_travelled=fly.vel*a
    net_counter_force_per_mass=np.linalg.norm(counter_acc)
    return net-counter_force_per_mass*distance_travelled

def epsilon(POS,SPEED,m1,m2,m3,m4): #m1,m2,m3,m4 are sun,earth,mars,venus, k is i (which element of array pointer)
    EK=0.5*SPEED**2
    r1=m1.pos-POS
    r2=m2.pos-POS
    r3=m3.pos-POS
    r4=m4.pos-POS
    magr1=np.linalg.norm(r1)
    magr2=np.linalg.norm(r2)
    magr3=np.linalg.norm(r3)
    magr4=np.linalg.norm(r4)
    ham=EK-G*((m1.mass/magr1)+(m2.mass/magr2)+(m3.mass/magr3)+(m4.mass/magr4))
    return ham

earth = Planet("earth", 0 , 147.92e+9,6371000,3.154e+7,0) #period 3.154e+7
mars = Planet("mars", 0, 218.49e+9, 3389500, 5.858e+7,0.875)#theoretical angular offset0.7768775717671333, period  59.4e+6
sun = Planet("sun", 1.99e+30, 0, 696340000, 31446925,0)
venus = Planet("venus", 0, 108.2e+9, 605200, 19440000,1.55334)


rocket = Satellite("rocket", 2.8e+6,np.array([earth.radius+earth.dist, 0]))
rocket_vel_at_pole=((earth.dist+earth.radius)/earth.period)*2*PI
rocket.vel[1]=(((earth.dist+earth.radius)/earth.period)+(earth.radius/86400))*2*PI

#create the arrays to be exported
speed=np.array([np.linalg.norm(rocket.vel)])
MX=np.array(mars.pos[0])
MY=np.array(mars.pos[1])
EX=np.array(earth.pos[0])
EY=np.array(earth.pos[1])
VX=np.array(venus.pos[0])
VY=np.array(venus.pos[1])
RX=np.array(rocket.pos[0])
RY=np.array(rocket.pos[1])
hams=np.array([epsilon(rocket.pos,speed[0],sun,earth,mars,venus)])
#----------------------

r1=earth.radius+earth.dist
u=sun.mass*G
r2=mars.dist
starting=KE_per_mass(rocket)
deltaV=np.sqrt(u/r1)*(np.sqrt(2*r2/(r1+r2))-1)  #theoretical from vis viva eqn
burn1=np.array([0,deltaV])
boost(rocket,burn1)
error=np.array([0,30])   #29.95 not enough, 30.0 too much
boost(rocket,error)
extra=(earth.radius/86400)*2*PI
burn1=burn1[1]+error[1]
mid=KE_per_mass(rocket)

#Create figure
fig=plt.figure(figsize=(12,12))
#Create 3D axes
ax=fig.add_subplot(111)

#append all so they are all the same length in the end
MX=np.append(MX,mars.pos[0])
MY=np.append(MY,mars.pos[1])
EX=np.append(EX,earth.pos[0])
EY=np.append(EY,earth.pos[1])
VX=np.append(VX,venus.pos[0])
VY=np.append(VY,venus.pos[1])
RX=np.append(RX,rocket.pos[0])
RY=np.append(RY,rocket.pos[1])
speed=np.append(speed,np.linalg.norm(rocket.vel))
hams=np.append(hams,epsilon(rocket.pos,speed[1],sun,earth,mars,venus))

np.savetxt('example.txt', np.c_[EX,EY,MX,MY,RX,RY], delimiter=',')

data=open('example.txt', 'ab')
#-----------------------
for i in range(0,n):
    earth.pos=earth.dist*np.array([np.cos(2*PI*dt*i/earth.period+earth.theta),np.sin(2*PI*dt*i/earth.period+earth.theta)])
    mars.pos=mars.dist*np.array([np.cos(2*PI*dt*i/mars.period+mars.theta),np.sin(2*PI*dt*i/mars.period+mars.theta)])
    venus.pos=venus.dist*np.array([np.cos(2*PI*dt*i/venus.period+venus.theta),np.sin(2*PI*dt*i/venus.period+venus.theta)])
    #print(rocket.vel)
    RK4_nbody(rocket, sun, earth, mars, venus, dt)
    
    #append all equally 
    
    RX=np.append(RX,rocket.pos[0])
    RY=np.append(RY,rocket.pos[1])
    MX=np.append(MX,mars.pos[0])
    MY=np.append(MY,mars.pos[1])
    EX=np.append(EX,earth.pos[0])
    EY=np.append(EY,earth.pos[1])
    VX=np.append(VX,venus.pos[0])
    VY=np.append(VY,venus.pos[1])
    speed=np.append(speed,np.linalg.norm(rocket.vel))
    hams=np.append(hams,epsilon(rocket.pos,speed[i+2],sun,earth,mars,venus))
    
    np.savetxt(data, np.c_[EX,EY,MX,MY,RX,RY], delimiter=',')
    
    
    dist=np.linalg.norm(rocket.pos)
    if mars.dist-dist<mars.radius:
        marsvel=np.array([0,-23434.8])
        starting2=KE_per_mass(rocket)
        print("velocity before landing=", rocket.vel)
        burn2=marsvel-rocket.vel
        rocket.vel+=burn2
        burn2mag=np.linalg.norm(burn2)
        print("deltaV2:", burn2mag, "extra m/s")
        ending2=KE_per_mass(rocket)
        print("extra energy=", ending2-starting2,"j/kg")
        print("i=", i)
        if np.linalg.norm(mars.pos-rocket.pos)<mars.radius:
            print("perfect arrival")
            break
        else:
            print("Landing successfull")
            
            #append all so same lenght arrays "landing appends"
            RX=np.append(RX,MX[i])
            RY=np.append(RY,MY[i]-mars.radius)
            MX=np.append(MX,mars.pos[0])
            MY=np.append(MY,mars.pos[1])
            EX=np.append(EX,earth.pos[0])
            EY=np.append(EY,earth.pos[1])
            VX=np.append(VX,venus.pos[0])
            VY=np.append(VY,venus.pos[1])
            speed=np.append(speed,np.linalg.norm(rocket.vel))
            hams=np.append(hams,epsilon(rocket.pos,speed[i+3],sun,earth,mars,venus))
            np.savetxt(data,np.c_[EX,EY,MX,MY,RX,RY], delimiter=',')
            #landing complete, rocket matching martian velocity 
            break
    if mars.pos[1]-1000<0:
        print("mars arrived first")
        break
data.close()
        