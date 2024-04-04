def plot(possitions,orientations,t): #creates and saves an individual frame of the current state of the system
    plt.clf()
    plt.figure(figsize=(10,10))
    for i in range(N):
        plt.arrow(possitions[i].real, possitions[i].imag, delta*orientations[i].real, delta*orientations[i].imag,head_width=delta)
        plt.arrow(possitions[i].real+L, possitions[i].imag, delta*orientations[i].real, delta*orientations[i].imag,head_width=delta)
        plt.arrow(possitions[i].real-L, possitions[i].imag, delta*orientations[i].real, delta*orientations[i].imag,head_width=delta)
        plt.arrow(possitions[i].real, possitions[i].imag+L, delta*orientations[i].real, delta*orientations[i].imag,head_width=delta)
        plt.arrow(possitions[i].real, possitions[i].imag-L, delta*orientations[i].real, delta*orientations[i].imag,head_width=delta)
    TITLE = "$t = $"+str(t)+"$\:\: \\varphi^t = $"+str(round(abs(np.mean(orientations)),3))
    plt.title(TITLE)
    plt.xlim(0,L)
    plt.ylim(0,L)
    NAME = str(t) + '.png'
    plt.savefig(NAME)
    plt.close()
    return
##---------------------------------------------------------------------------------------------------------------------------------
def generate_starter_possitions():
    possitions = np.zeros(N,dtype=complex)
    for i in range(N):
        possitions[i] = random.uniform(0,L)+random.uniform(0,L)*1j
    return possitions
##---------------------------------------------------------------------------------------------------------------------------------
def generate_starter_orientations():
    orientations = np.zeros(N,dtype=complex)
    for i in range(N):
        orientations[i] = cmath.exp(1j*random.uniform(0,2*cmath.pi))
    return orientations
##---------------------------------------------------------------------------------------------------------------------------------
def update_possitions(possitions,orientations):
    new_possitions = np.zeros(N,dtype=complex)
    for i in range(N):
        new_possitions[i] = possitions[i] + v_0*orientations[i]
        new_possitions[i] = (new_possitions[i].real % L) +1j*(new_possitions[i].imag % L)
    return new_possitions
##---------------------------------------------------------------------------------------------------------------------------------
def update_orientations(possitions,orientations):
    new_orientations = np.zeros(N,dtype=complex)

    for i in range(N):
        interaction = 0
        for j in range(N):
            x_seperation = abs(possitions[i].real-possitions[j].real)
            if x_seperation > L/2:
                x_seperation = L - x_seperation
            y_seperation = abs(possitions[i].imag-possitions[j].imag)
            if y_seperation > L/2:
                y_seperation = L - y_seperation
            if x_seperation**2 + y_seperation**2 < r_0**2:
                interaction += orientations[j]
        new_orientations[i] = cmath.exp(1j*random.uniform(-n*cmath.pi,n*cmath.pi))*interaction/abs(interaction)
    return new_orientations
##---------------------------------------------------------------------------------------------------------------------------------
def makemydir(whatever): #makes a folder
  try:
    os.makedirs(whatever)
  except OSError:
    pass
  os.chdir(whatever)
##---------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import cmath
import math
import random
import matplotlib.pyplot as plt
import os
import sys
import cv2
import glob

##Folder Admin
makemydir("plots")
if os.path.isdir("current"):
    sys.exit("Danger of Overwrite. Rename 'current' folder.")
else:
    makemydir("current")


T=500 #length of simulation
L = 10 #low density = 100; high density = 10
N = 400
delta = 0.01*L #just a plotting variable
v_0 = 0.5
n =0.5 #eta: low = 0.1; high = 0.5
r_0 = 1

##Create a text file with parameters
f = open("00parameters.txt", "x")
f.write("L = " + str(L))
f.write("\nN = " + str(N))
f.write("\nv_0 = " + str(v_0))
f.write("\neta = " + str(n))
f.write("\nr_0 = " + str(r_0))
f.close()

##run simulation
print("0/"+str(T))
possitions = generate_starter_possitions()
orientations = generate_starter_orientations()
plot(possitions,orientations,0)
order_param = np.array([abs(np.mean(orientations))])
for t in range(1,T+1):
    print(str(t)+"/"+str(T))
    possitions = update_possitions(possitions,orientations)
    orientations = update_orientations(possitions,orientations)
    plot(possitions,orientations,t)
    order_param = np.append(order_param,[abs(np.mean(orientations))])

##create plot of order parameter
plt.clf()
plt.plot(range(len(order_param)),order_param)
plt.xlabel("$t$")
plt.ylabel("$\\varphi^t$")
plt.ylim(0,1)
plt.xlim(0,len(order_param)-1)
NAME = '00order.png'
plt.savefig(NAME)
plt.close()

#create film
img_array = []
for n in range(T+1):
    filename = str(n) + ".png"
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
out = cv2.VideoWriter('00video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
