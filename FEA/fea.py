from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from mshr import *
import sys
import time
#-------------------------------------Define Function for FEA---------------------------------------------------------#
def hyp_postbuck(L,w,E,nu,P,f_name,disp_ini,disp_fin,d_disp_inc):

    quad_flag = P

    parameters["form_compiler"]["representation"] = "uflacs"
    parameters["form_compiler"]["cpp_optimize"] = True #tells form compiler to use C++ compiler
    ffc_options = {"optimize": True, # turns on optimization for code generation
                "eliminate_zeros": True, # Basis function values will be compressed so they only contain non-zero values for less # of operations; overhead introduced for mapping
                "precompute_basis_const": True, #Precompute constant terms in basis 
                "precompute_ip_constant": True} #same but for integration points
    if quad_flag == 1:
        parameters["form_compiler"]["quadrature_degree"] = 1
    if quad_flag == 2:
        parameters["form_compiler"]["quadrature_degree"] = 2
        
   

    #--------------------------------Meshing---------------------------------------#

    mesh = Mesh(f_name)
    #-----------------------------Choose Element Type-------------------------------#

    if quad_flag == 1:
        P2 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    if quad_flag == 2:
        P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    TH = P2

    W = FunctionSpace(mesh, TH)
    #----------------------------Fixed Boundary Conditions--------------------------#

    btm  =  CompiledSubDomain("on_boundary && near(x[1], btmCoord)", btmCoord = 0)
    btmBC = DirichletBC(W, Constant((0.0,0.0)), btm)   #Fix bottom in y direction
    top  =  CompiledSubDomain("on_boundary && near(x[1], topCoord)", topCoord = L)
    topBCX = DirichletBC(W.sub(0), Constant(0.0),top)  #Fix x portion of top


    #----------------------------Forces--------------------------------------------#
   
    T  = Constant((0.0, 0.0))  # Traction force on the boundary
    B  = Constant((0.0, 0.0))  # Body Force

    #----------------------------------Applied Displacement---------------------#
    disp = Expression("-(disp_ini + d_disp)",d_disp = 0.0,disp_ini = 0.0, degree = 1)
    topBCY = DirichletBC(W.sub(1),disp,top)
    bcs = [btmBC, topBCY, topBCX]
    #--------------------------Test and Trial Functions-----------------------------#

    u = Function(W)
    du = TrialFunction(W)
    v = TestFunction(W)

    #-----------------------------Material Parameters-------------------------------#

    lmbda, mu = (E*nu/((1.0 + nu )*(1.0-2.0*nu))) , (E/(2.0*(1.0+nu)))
    #----------------------------------Kinematics------------------------------------#

    d = len(u)
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    F = variable(F)

    #-----------------------------------Hyperelastic Free energy----------------------------#

    psi = 1/2*mu*( inner(F,F) - 3 - 2*ln(det(F)) ) + 1/2*lmbda*(1/2*(det(F)**2 - 1) - ln(det(F)))
    f_int = derivative(psi*dx,u,v)
    f_ext = derivative( dot(B, u)*dx('everywhere') + dot(T, u)*ds('everywhere') , u, v)
    #-----------------------------------Total potential energy-------------------------------#

    Fboth = f_int - f_ext

    #-----------------------------------Jacobian/Tangential Stiffness Matrix-------------------#
    dF = derivative(Fboth, u, du)

    #----------------------------------Displacement Increment-----------------------------------#
    d_disp = 0.0

    ii = 0
    Fy = []
    u_b = []
    eig_val = []
    
    u_x = [0,0]
    # Stop simulation once x-displacement reaches threshold or column is over Euler load
    while disp_ini + d_disp - d_disp_inc <= disp_fin and max([abs(max(u_x)),abs(min(u_x))]) < 0.15*w:
        #--------------------------------------------Update Displacement-------------------------------------#
        disp.d_disp = d_disp
        # ------------------------------Solve for displacement------------------------------------#
        solve(Fboth == 0, u, bcs, J = dF,form_compiler_parameters = ffc_options)
    
    
  
        #---------------------------------------------Reaction Force Calculations--------------------------#
    
        y_dof = W.sub(1).dofmap().dofs()
        f_react = assemble(f_int) - assemble(f_ext) # Nodal Reaction forces
    
        dof_coord = W.tabulate_dof_coordinates().reshape((-1,2))
        y_bot = np.min(dof_coord[:,1]) + 10e-5
        y_bot_dof = []
        for jj in y_dof:
            if dof_coord[jj,1] < y_bot:
                y_bot_dof.append(jj)

        Fy.append(np.sum(f_react[y_bot_dof]))
        
        #-------------------------------------Find left/right symmetry-----------------------------------------#
        x_dof = W.sub(0).dofmap().dofs()
        u_xy = np.array(u.vector())
        u_x = u_xy[x_dof]

        if abs(max(u_x)) > abs(min(u_x)):
          symm = 1
        else:
          symm = 0
    
        print('|Displacement',disp_ini+d_disp,'|Reaction Force',Fy[ii],'|')
       
        d_disp +=d_disp_inc
       
        ii+=1
        
    return Fy,u,disp_ini + d_disp - d_disp_inc,symm,eig_val

#-----------------------------OS Commands------------------------------------#
num = int(sys.argv[1]) # array in batch
f_mesh = 'mesh/mesh'+str(num)+'.xml' # Directory to mesh
f_lr = 'lr/lr.txt' # Directory to .txt that save results

#----------------------Beam and Material Parameters------------------------------------------------#
L = 40.0 # 40.0 for subdataset1; 800.0 for subdataset2 and subdataset 3
w = 5.0 # 5.0 for subdataset1; 100.0 for subdataset2 and subdataset 3
b = 1
E = 1.0 # Young's Modulus
nu = 0.3 # Poisson's Ratio
P = 2
disp_ini = 0.0
d_disp_inc = (2.5e-4)*L # incremental displacement
#------------------------Set final displacement as Euler buckling for beam with no holes with scaling factor----------------#
k_factor = 0.5
I = w**3/12.0

P_cr = np.pi**2.0 * E * I / (k_factor * L)**2.0
sig_cr = P_cr / (b * w)
eps_cr = sig_cr / E
disp_cr = eps_cr * L
disp_fin = disp_cr * 1.05
#------------------------------Solve FEA Problem-------------------------------#
ti = time.time()
sol = hyp_postbuck(L,w,E,nu,P,f_mesh,disp_ini,disp_fin,d_disp_inc)
tf = time.time()

print('Time:' ,tf-ti,'seconds')
lr = sol[-2]
eig_val = sol[-1]
Fy = sol[0]
print(Fy)
print(lr,'Line:',num,'Batch:',batch)

#-----------------------Save deformation plot-------------------------------------------#
# Uncomment to see displacement profile.

#u = sol[1]
#u_name = 'u/u'+str(mesh_num)+'.pvd' # filename for displacement plot
#file = File(u_name)
#file << u
