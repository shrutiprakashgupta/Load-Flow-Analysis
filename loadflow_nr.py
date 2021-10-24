import pandas as pd
import numpy as np
import argparse

#Command
#python gauss_seidel.py -ld linedata.csv -bd busdata.csv
parser = argparse.ArgumentParser()
parser.add_argument("-ld", "--linedata", help = "Csv file with Linedata")
parser.add_argument("-bd", "--busdata", help = "Csv file with Busdata")
args = parser.parse_args()

#Reading line data from csv file 
linedata = pd.read_csv(args.linedata)

#Reading bus data from csv file 
busdata = pd.read_csv(args.busdata)

#Resistance and Impedance matrices
R       = np.array(linedata["R"],           dtype=float)
X       = np.array(linedata["X"],           dtype=float)
B       = np.array(1j * linedata["Y_2"],    dtype=complex)
Z       = np.array(R + 1j * X,              dtype=complex)
Y       = 1/Z
nline   = len(linedata)
nbus    = len(busdata)

#Creating the Ybus matrix 
From    = np.array(linedata['From'],    dtype=int) - 1
To      = np.array(linedata['To'],      dtype=int) - 1
Ybus    = np.zeros([nbus, nbus],        dtype=complex)
for i in range(nline):
    Ybus[From[i],To[i]]     = Ybus[From[i],To[i]]   - Y[i]
    Ybus[To[i],From[i]]     = Ybus[From[i],To[i]]
    Ybus[From[i],From[i]]   = Ybus[From[i],From[i]] + Y[i] + B[i]
    Ybus[To[i],To[i]]       = Ybus[To[i],To[i]]     + Y[i] + B[i]

G       = Ybus.real
B       = Ybus.imag

#Reading the power and votlage specifications of the buses 
bus_type = np.array(busdata['Bus Type'])
Pg      = np.array(busdata['Pg'],       dtype=float)
Qg      = np.array(busdata['Qg'],       dtype=float)
Pd      = np.array(busdata['Pd'],       dtype=float)
Qd      = np.array(busdata['Qd'],       dtype=float)
Vmag    = np.array(busdata['|V|'],      dtype=float)
delta   = np.array(busdata['delta'],    dtype=float)
Qmin    = np.array(busdata['Qmin'],     dtype=float)
Qmax    = np.array(busdata['Qmax'],     dtype=float)
pq_bus  = []
i       = 0
for t in bus_type:
    if(t == 3):
        pq_bus.append(i)
    i = i + 1
n0      = len(pq_bus)

#Initialising power and voltage values 
V   = Vmag * (np.cos(delta) + 1j * np.sin(delta))
P   = Pg - Pd
Q   = Qg - Qd
error = 1

i   = 1
max_iter    = 50
err_threshold   = 0.000001

while ((i < max_iter) and (error > err_threshold)):
    
    #P and Q calculation
    Scalc = np.multiply(np.conjugate(V), np.matmul(Ybus,V))
    Pcalc = Scalc.real
    Qcalc = -1*Scalc.imag

    #Jacobian matrix formation
    J11 = np.zeros([nbus-1,nbus-1])
    for n in range(0,nbus-1):
        for m in range(0,nbus-1):
            if(n != m):
                J11[n,m] = -1 * abs(Ybus[n+1,m+1]) * Vmag[n+1] * Vmag[m+1] * np.sin(np.angle(Ybus[n+1,m+1]) + delta[m+1] - delta[n+1])
            else:
                J11[n,m] = -1*Qcalc[n+1] - (Vmag[n+1] * Vmag[n+1] * B[n+1,n+1]) 
    
    J21 = np.zeros([n0,nbus-1])
    for n in range(0,n0):
        for m in range(0,nbus-1):
            bn = pq_bus[n]
            if(bn != m+1):
                J21[n,m] = -1 * abs(Ybus[bn,m+1]) * Vmag[bn] * Vmag[m+1] * np.cos(np.angle(Ybus[bn,m+1]) + delta[m+1] - delta[bn])
            else:
                J21[n,m] = Pcalc[bn] - (Vmag[bn] * Vmag[bn] * G[bn,bn])

    J12 = np.zeros([nbus-1,n0])
    for n in range(0,nbus-1):
        for m in range(0,n0):
            bm = pq_bus[m]
            if(bm != n+1):            
                J12[n,m] = abs(Ybus[n+1,bm]) * Vmag[n+1] * Vmag[bm] * np.cos(np.angle(Ybus[n+1,bm]) + delta[bm] - delta[n+1])
            else:
                J12[n,m] = Pcalc[bm] + (Vmag[bm] * Vmag[bm] * G[bm,bm])

    J22 = np.zeros([n0,n0])
    for n in range(0,n0):
        for m in range(0,n0):
            bn = pq_bus[n]
            bm = pq_bus[m]
            if(bn != bm):
                J22[n,m] = -1 * abs(Ybus[bn,bm]) * Vmag[bn] * Vmag[bm] * np.sin(np.angle(Ybus[bn,bm]) + delta[bm] - delta[bn])
            else:
                J22[n,m] = Qcalc[bn] - (Vmag[bn] * Vmag[bn] * B[bn,bn]) 

    J = np.zeros([nbus+n0-1,nbus+n0-1], dtype=complex)
    J[:nbus-1,:nbus-1]  = J11
    J[:nbus-1,nbus-1:]  = J12
    J[nbus-1:,:nbus-1]  = J21
    J[nbus-1:,nbus-1:]  = J22

    #Calculating difference with updated value
    dP              = P - Pcalc
    dQ              = Q - Qcalc
    dPQ             = np.zeros(nbus+n0-1)
    dPQ[:nbus-1]    = dP[1:]
    n = 0
    m = 0
    for n in range(nbus):
        if(bus_type[n] == 3):
            dPQ[nbus-1+m] = dQ[n]
            m = m + 1

    #Correction in the Voltage magnitudes and angles 
    corr = np.matmul( np.linalg.inv(J), dPQ)
    corr = corr.real
    for n in range(1,nbus):
        delta[n]        = delta[n] + corr[n-1]
    for n in range(n0):
        Vmag[pq_bus[n]] = Vmag[pq_bus[n]] *(1 + corr[nbus-1+n])
    
    dV      = Vmag * (np.cos(delta) + 1j*np.sin(delta)) - V
    V       = Vmag * (np.cos(delta) + 1j * np.sin(delta))
    
    error   = np.linalg.norm(dV)
    i       = i + 1

comp_data = np.zeros([nbus,10])

comp_data[:,0] = Vmag
comp_data[:,1] = delta*(180/(22/7))

#Given data
comp_data[:,2] = Pg
comp_data[:,3] = Qg
comp_data[:,4] = Pd
comp_data[:,5] = Qd

#Power calculations at each bus
S = np.multiply(np.conjugate(V), np.matmul(Ybus, V))
comp_data[:,6] = S.real
comp_data[:,7] = -1*S.imag

#Power loss calculations at each bus
L = (Pg - Pd) - 1j*(Qg - Qd) - S
comp_data[:,8] = L.real
comp_data[:,9] = L.imag

power_flow = pd.DataFrame(data=comp_data, index=np.arange(nbus), columns=['Vmag','Vang','Pg','Qg','Pd','Qd','Pcalc','Qcalc','Ploss','Qloss'])

print("Load flow analysis with Newton Raphson Method")
print("# iter for convergence: " + str(i))
print("Error: " + str(error))
print(power_flow[['Vmag','Vang']])
print(power_flow[['Pg', 'Qg', 'Pd', 'Qd']])
print(power_flow[['Pcalc', 'Qcalc', 'Ploss', 'Qloss']])
