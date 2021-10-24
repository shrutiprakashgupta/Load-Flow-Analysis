import pandas as pd
import numpy as np
import argparse

#Command
#python gauss_seidel.py -ld linedata.csv -bd busdata.csv
parser = argparse.ArgumentParser()
parser.add_argument("-ld", "--linedata", help = "Csv file with Linedata")
parser.add_argument("-bd", "--busdata", help = "Csv file with Busdata")
args = parser.parse_args()

#scaling factor for faster convergence of the algorithm 
alpha   = 1.6

#Reading line data from csv file 
linedata = pd.read_csv(args.linedata)

#Reading bus data from csv file 
busdata = pd.read_csv(args.busdata)

#Resistance and Impedance matrices
R       = np.array(linedata["R"], dtype=float)
X       = np.array(linedata["X"], dtype=float)
B       = np.array(1j * linedata["Y_2"], dtype=complex)
Z       = np.array(R + 1j * X, dtype=complex)
Y       = 1/Z
nline   = len(linedata)
nbus    = len(busdata)

#Creating the Ybus matrix 
From    = np.array(linedata['From'], dtype=int) - 1
To      = np.array(linedata['To'], dtype=int) - 1
Ybus    = np.zeros([nbus, nbus], dtype=complex)
for i in range(nline):
    Ybus[From[i],To[i]]     = Ybus[From[i],To[i]] - Y[i]
    Ybus[To[i],From[i]]     = Ybus[From[i],To[i]]
    Ybus[From[i],From[i]]   = Ybus[From[i],From[i]] + Y[i] + B[i]
    Ybus[To[i],To[i]]       = Ybus[To[i],To[i]] + Y[i] + B[i]

#Reading the power and votlage specifications of the buses 
bus_type = np.array(busdata['Bus Type'])
Pg      = np.array(busdata['Pg'])
Qg      = np.array(busdata['Qg'])
Pd      = np.array(busdata['Pd'])
Qd      = np.array(busdata['Qd'])
Vmag    = np.array(busdata['|V|'])
delta   = np.array(busdata['delta'])
Qmin    = np.array(busdata['Qmin'])
Qmax    = np.array(busdata['Qmax'])

#Initialising power and voltage values 
V   = Vmag * (np.cos(delta) + 1j * np.sin(delta))
P   = Pg - Pd
Q   = Qg - Qd
error = 1

i   = 1
max_iter    = 50
err_threshold   = 0.000001

while ((i < max_iter) and (error > err_threshold)):
    #Bus 1 is kept to be the slack bus thus no calculation required for it 
    
    dV  = np.zeros(nbus, dtype=complex)
    for n in range(1,nbus):
        
        if(bus_type[n] == 2): 
        #PV bus 
            I   = sum(np.multiply(Ybus[n,:],V))
            S   = np.conjugate(V[n]) * I
            Q[n] = -1*S.imag

            A   = I - Ybus[n,n]*V[n]

            if((Q[n] > Qmin[n]) and (Q[n] < Qmax[n])) : 
                Vnew    = (1/Ybus[n,n]) * ((P[n] - 1j*Q[n])/np.conjugate(V[n]) - A)
                Vnew    = Vmag[n] * (Vnew / abs(Vnew))
                #Magnitude correction
                dV[n]   = Vnew - V[n]
                V[n]    = Vnew
            
            else:
            #Treated as PQ bus
                if(Q[n] < Qmin[n]): 
                    Q[n] = Qmin[n]
                elif(Q[n] > Qmax[n]):
                    Q[n] = Qmax[n]
            
                Vnew    = (1/Ybus[n,n]) * ((P[n] - 1j*Q[n])/np.conjugate(V[n]) - A)
                Vnew    = V[n] + alpha * (Vnew - V[n])
                dV[n]   = Vnew - V[n]
                V[n]    = Vnew

        elif(bus_type[n] == 3):
        #PQ bus
            I = sum(np.multiply(Ybus[n,:],V))
            A = I - Ybus[n,n]*V[n]

            Vnew    = (1/Ybus[n,n]) * ((P[n] - 1j*Q[n])/np.conjugate(V[n]) - A)
            Vnew    = V[n] + alpha * (Vnew - V[n])
            dV[n]   = Vnew - V[n]
            V[n]    = Vnew

    error = np.linalg.norm(dV)
    i = i + 1

comp_data = np.zeros([nbus,10])

comp_data[:,0] = np.abs(V)
comp_data[:,1] = np.angle(V, deg=True)

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

print("Load flow analysis with Gauss Seidel Method")
print("# iter for convergence: " + str(i))
print("Error: " + str(error))
print(power_flow[['Vmag','Vang']])
print(power_flow[['Pg', 'Qg', 'Pd', 'Qd']])
print(power_flow[['Pcalc', 'Qcalc', 'Ploss', 'Qloss']])
