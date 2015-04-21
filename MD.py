#Python: Monte Carlo Simulation by Z. Yao
import matplotlib 
import numpy as nm
import matplotlib.pyplot as plt
#    Function_1: Energy Calculation
def Energy(M,V1,V2,u):
    M_Left       =  nm.roll(M      ,-1,axis=1)        #Periodic boundary condition
    M_Right      =  nm.roll(M      , 1,axis=1)
    M_Up         =  nm.roll(M      ,-1,axis=0)
    M_Down       =  nm.roll(M      , 1,axis=0)
    M_Left_Up    =  nm.roll(M_Left ,-1,axis=0)
    M_Left_Down  =  nm.roll(M_Left , 1,axis=0)
    M_Right_Up   =  nm.roll(M_Right,-1,axis=0)
    M_Right_Down =  nm.roll(M_Right, 1,axis=0)        #Applying ising model to calculate energy
    E            =  0.5*V1*(M*M_Left+M*M_Right+M*M_Up+M*M_Down)+0.5*V2*(M*M_Left_Up+M*M_Right_Up+M*M_Left_Down+M*M_Right_Down)-u*(M)
    return nm.sum(E)
#    Function_2: Heat Capacity Calculation
def Cv(M,V1,V2,u,T):
    N            =  nm.size(M)
    M_Left       =  nm.roll(M      ,-1,axis=1)        #Periodic boundary condition
    M_Right      =  nm.roll(M      , 1,axis=1)
    M_Up         =  nm.roll(M      ,-1,axis=0)
    M_Down       =  nm.roll(M      , 1,axis=0)
    M_Left_Up    =  nm.roll(M_Left ,-1,axis=0)
    M_Left_Down  =  nm.roll(M_Left , 1,axis=0)
    M_Right_Up   =  nm.roll(M_Right,-1,axis=0)
    M_Right_Down =  nm.roll(M_Right, 1,axis=0)        #Applying ising model to calculate energy
    Energy       =  0.5*V1*(M*M_Left+M*M_Right+M*M_Up+M*M_Down)+0.5*V2*(M*M_Left_Up+M*M_Right_Up+M*M_Left_Down+M*M_Right_Down)-u*(M)
    C            =  (nm.sum(Energy**2)/N-(nm.sum(Energy)/N)**2)/T**2        #Calculate heat capacity
    return C
#    Function_3: Monte Carlo Step
def MCStep(M,V1,V2,u,T):
    for N_sites in range(0,nm.size(M)):
        RND      =  nm.random.rand()                                 #Random number x in (0,1)
        i        =  nm.random.choice(range(0,nm.shape(M)[1]))        #Randomly select one spin to flip
        j        =  nm.random.choice(range(0,nm.shape(M)[1]))
        M_i      =  nm.copy(M)
        M_f      =  nm.copy(M)
        M_f[i,j] =  (-1)*M_i[i,j]
        E_i      =  Energy(M_i,V1,V2,u)                              #Metropolis algorithm
        E_f      =  Energy(M_f,V1,V2,u)
        Diff_E   =  E_f-E_i
        P        =  nm.exp(-Diff_E/T)
        if P     >  RND:
            M    =  nm.copy(M_f)
        else:
            M    =  nm.copy(M_i)
        return M
#    Main Code
Dim    =    70        #System dimension
V1     =   1.0        #Reduced units: Chemical potential unit: m/|V1|; Temperature unit: kT/|V1|
V2     =  -2.0
N_Equi = 60000        #Step number for equlibrating
N_Samp = 40000        #Step number for sampling
u_max  =    10        #Chemical potential range
u_min  =   -10
u_step =   0.4
T_max  =    10        #Temperature range
T_min  =     0
T_step =   0.5
Matrix =  nm.ones([Dim,Dim])        #System definition
for T in nm.arange(T_min,T_max+T_step,T_step):
    S_Cv_vs_u=open('S_Cv_vs_u for T = %s.csv'%T,'w')
    U      =    []
    S      =    []
    C      =    []
    for u in nm.arange(u_min,u_max+u_step,u_step):
        Sigma_i       =  [nm.sum(Matrix)/nm.size(Matrix)]       
        C_i           =  [Cv(Matrix,V1,V2,u,T)]
        for N_steps in range(0,N_Equi):                          #Equilibrating runs
            Matrix    =  nm.copy(MCStep(Matrix,V1,V2,u,T))
            Sigma_i.append(nm.sum(Matrix)/(nm.size(Matrix)))
            C_i.append(Cv(Matrix,V1,V2,u,T))
        Sigma_s       =  []
        C_s           =  []
        for N_steps in range(0,N_Samp):                          #Sampling runs
            Matrix    =  nm.copy(MCStep(Matrix,V1,V2,u,T))
            Sigma_s.append(nm.sum(Matrix)/(nm.size(Matrix)))
            C_s.append(Cv(Matrix,V1,V2,u,T))
        S_avg         =  nm.sum(Sigma_s)/len(Sigma_s)            #Averaging
        C_avg         =  nm.sum(C_s)/len(C_s)
        S_Cv_vs_u.write("%s,%s,%s\n"%(u,S_avg,C_avg))
        U.append(u)
        S.append(S_avg)
        C.append(C_avg)    
    S_Cv_vs_u.close()
    fig,ax1 =  plt.subplots()                                   #Plotting
    ax2     =  ax1.twinx()
    ax1.plot(U,S,'b-s')
    ax2.plot(U,C,'r-s')                                        
    ax1.set_xlabel(r'Chemical Potential $\mu$')
    ax1.set_xlim(-10,10)
    ax1.set_ylabel(r'Composition <$\sigma$>')
    ax2.set_ylabel(r'Heat Capacity Cv')
    plt.savefig('Composition_Cv_u Temperature=%s.png'%T)
    plt.close()
#    Code End
