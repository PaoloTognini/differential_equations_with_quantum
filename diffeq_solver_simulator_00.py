import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, solve

# Definition of variables
n = 8    # number of copies. We need n >= 2
assert(n >= 2)
T = 0.01    # time of simulation
dt = 0.001    # time step
nT = int(T/dt)    # number of time steps

x_0 = 0.6    # initial condition
M = 1.    # upper bound on the function

print("T = ", T, ", dt = ", dt, ", x_0 = ", x_0, ", M = ", M)

solutions1 = [x_0]
pred = [x_0]
discrete_sol = [x_0]
norm_pred = [x_0]
solutions5 = [1]
solutions6 = [1]
purities = [1]
c1s = [0]
c2s = [0]
eigenvalue_max = [1]

# Results for the code which found the wrong eigenvector
# n = 2: M = 1.0003 (vaguely)
# n = 3: M = 1.015
# n = 4: M = 1.15
# n = 5: M = 1.3

b = 1/np.sqrt(2)
a = 1/(M * np.sqrt(2))
def gamma(x):
    return np.sqrt(1 - b*b - a*a*x*x)

c_1 = 1.
c_2 = -0.5
#define a vector in numpy with components b, a*x, gamma
def phi(x):
    return np.array([b, a*x, gamma(x)])
phi_0 = phi(x_0)

eigenvector_max_0 = [phi_0[0]]
eigenvector_max_1 = [phi_0[1]]
eigenvector_max_2 = [phi_0[2]]

# define a recursive function that computes the tensor product of Psi with phi(x)
def tensor_exp(phi, n):
    assert(n >= 0)
    if n == 0:
        return np.array([1])
    else:
        return np.tensordot(tensor_exp(phi, n-1), phi, axes=0)

# define the initial state
Psi_0 = tensor_exp(phi_0, n)

# now transform Psi_0 into a vector
Psi_0 = np.reshape(Psi_0, (3**n))

# define the hamiltonian A with a for loop on the copies.
A = np.zeros((3**n, 3**n))
B = np.zeros((3**n, 3**n))

print("phi_0 = ", phi_0)

for i in range(n):
    for j in range(n):
        # write a for loop that iterates over all numbers from 0 to 3**n, but with the i-th and j-th digit
        # in trinary (base 3) equal to 0.
        if i != j:
            for k in range(3**n):
                i_digit = (k//3**i)%3
                j_digit = (k//3**j)%3
                #A[k, k] = 1/(2*(n-1))
                if i_digit == 1 and j_digit == 1:
                    print("At k = ", k, " we have: i_digit = ", i_digit, "j_digit = ", j_digit)
                    A[k, k] -= 1./(2*(n-1)*c_1*a*a)    #A_2[11_3, 11_3] where the footnote means base 3
    for k in range(3**n):
        i_digit = (k//3**i)%3
        if i_digit == 0:
            B[2*3**i + k, k] = 1.   #A_2[2_3, 0_3] where the footnote means base 3

# do the temporal evolution

Psi = Psi_0
x_2 = x_0
x_3 = x_0
x_4 = x_0
for nt in range(nT):
    print("nt = ", nt)
    print("Psi_0.shape = ", Psi_0.shape)
    print("Psi.shape = ", Psi.shape)
    print("A.shape = ", A.shape)

    # evolve the state applying matrix A onto Psi
    print("A = ", A)
    Psi_A = Psi + np.dot(A, Psi)*dt    # Psi evolved with A alone (not with B)
    print("difference between Psi and Psi_A = ", Psi - Psi_A)
    Psi_B = np.dot(B, Psi)*dt    # contribution to the evolution given by B


    # To impose norm of Psi_A + Psi_B*c equal to 1, we must solve a second degree equation for c:
    a_eq = np.dot(Psi_B, Psi_B)
    b_eq = 2*np.dot(Psi_A, Psi_B)
    c_eq = np.dot(Psi_A, Psi_A) - 1.
    delta_eq = b_eq**2 - 4*a_eq*c_eq

    print("a_eq, b_eq, c_eq = ", a_eq, b_eq, c_eq)
    print("delta_eq = ", delta_eq)
    if delta_eq < 0:
        break

    c_calculated1 = -b_eq/(2*a_eq) + np.sqrt(delta_eq)/(2*a_eq)
    c_calculated2 = -b_eq/(2*a_eq) - np.sqrt(delta_eq)/(2*a_eq)
    c1s.append(c_calculated1)
    c2s.append(c_calculated2)
    # choose the solution with the smallest absolute value
    
    if abs(c_calculated1) < abs(c_calculated2):
        c_calculated = c_calculated1
        print("c_calculated1 = ", c_calculated1, " at nt = ", nt, " (time = ", nt*dt, ").")
    else:
        c_calculated = c_calculated2
        print("c_calculated2 = ", c_calculated2, " at nt = ", nt, " (time = ", nt*dt, ").")

    #c_calculated = c_calculated2

    #c_calculated = -np.dot(Psi, Psi_A)/np.dot(Psi, Psi_B)

    c_analytical = -a*a*x_2*x_2*x_2*x_2/(c_1*b*b*gamma(x_2)*gamma(x_2))
    print("c_analytical = ", c_analytical)
    print("c_calculated = ", c_calculated)

    # define Psi_0 as Psi where c is substituted with c_0
    
    Psi = Psi_A + c_calculated*Psi_B
    print("Norm of Psi = ", np.dot(Psi, Psi), " at nt = ", nt, " (time = ", nt*dt, ").")
    print("differenze between Psi and Psi_0 = ", Psi - Psi_0)

    # first print of the solution, wrong but simpler version. We do not print it anymore, as it is constant.
    x_1 = Psi[1]/(b**(n-1)*a)
    print("Psi_B = ", Psi_B)
    print("x_1 = ", x_1)

    solutions1.append(x_1)

    # second print of the solution, correct but more complicated version.
    # the first step is to transform Psi into a matrix with an index of dimension 3 and one of dimension
    # 3**(n-1)
    Psi_reshaped = np.reshape(Psi, (3, 3**(n-1)))
    print("Psi_reshaped.shape = ", Psi_reshaped.shape)
    # the second step is to calculate the density matrix of Psi
    # calculate the density matrix which will have dimensions (3, 9, 3, 9)
    density_matrix = np.tensordot(Psi_reshaped, Psi_reshaped, axes=0)
    print("density_matrix.shape = ", density_matrix.shape)
    # the third step is to do the trace of density_matrix with respect to the indices of dim 3**(n-1)
    # to get the reduced density matrix.
    # Calculate the reduced density matrix which will have dimensions (3, 3)
    reduced_density_matrix = np.trace(density_matrix, axis1=1, axis2=3)
    print("reduced_density_matrix.shape = ", reduced_density_matrix.shape)
    # we calculate purity
    purity = np.trace(np.dot(reduced_density_matrix, reduced_density_matrix))
    print("purity = ", purity, " at nt = ", nt, " (time = ", nt*dt, ").")
    purities.append(purity)
    # the fourth step is to calculate the eigenvector associated to the maximum eigenvalue of the reduced
    # density matrix
    eigenvector = np.linalg.eig(reduced_density_matrix)
    print("eigenvector = ", eigenvector)
    # print the eigenvalue with maximum absolute value
    print("max eigenvalue = ", np.max(eigenvector[0]))
    # print the eigenvector correspondent to the maximum eigenvalue
    new_phi_component0 = eigenvector[1][0][np.argmax(eigenvector[0])]
    new_phi_component1 = eigenvector[1][1][np.argmax(eigenvector[0])]
    new_phi_component2 = eigenvector[1][2][np.argmax(eigenvector[0])]

    # The eigenvector must be chosen with the convention that the 0-th component is positive.
    change_sign = np.sign(new_phi_component0)
    new_phi_component0 = new_phi_component0 * change_sign
    new_phi_component1 = new_phi_component1 * change_sign
    new_phi_component2 = new_phi_component2 * change_sign

    print(" DEBUG 1: New solution: ", new_phi_component0, new_phi_component1, new_phi_component2)
    sqrt_2 = 1.4142
    print(" DEBUG 2: Solution multiplied by sqrt(2): ", new_phi_component0*sqrt_2, new_phi_component1*sqrt_2, new_phi_component2*sqrt_2)

    #print("eigenvector of max eigenvalue = ", eigenvector[1][1][np.argmax(eigenvector[0])], " at nt = ", nt, " (time = ", nt*dt, ").")
    #eigenvector_max_0.append(eigenvector[1][np.argmax(eigenvector[0]), 0] * np.sign(eigenvector[1][np.argmax(eigenvector[0]), 0]))
    #eigenvector_max_1.append(eigenvector[1][np.argmax(eigenvector[0]), 1] * np.sign(eigenvector[1][np.argmax(eigenvector[0]), 0]))
    #eigenvector_max_2.append(eigenvector[1][np.argmax(eigenvector[0]), 2] * np.sign(eigenvector[1][np.argmax(eigenvector[0]), 0]))
    eigenvector_max_0.append(new_phi_component0)
    eigenvector_max_1.append(new_phi_component1)
    eigenvector_max_2.append(new_phi_component2)

    # print the eigenvalue with maximum absolute value
    print("max eigenvalue = ", np.max(eigenvector[0]))
    eigenvalue_max.append(np.max(eigenvector[0]))

    # x_2 is the component of index 1 of the eigenvector with maximum eigenvalue, divided by a
    #x_2 = eigenvector[1][np.argmax(eigenvector[0]), 1]/a * np.sign(eigenvector[1][np.argmax(eigenvector[0]), 0])
    #pred.append(x_2)
    #x_4 = eigenvector[1][np.argmax(eigenvector[0]), 1]*M/eigenvector[1][np.argmax(eigenvector[0]), 0]
    #norm_pred.append(x_4)
    #x_5 = abs(eigenvector[1][np.argmax(eigenvector[0]), 2]/a * np.sign(eigenvector[1][np.argmax(eigenvector[0]), 0]))
    #solutions5.append(x_5)
    #x_6 = abs(eigenvector[1][np.argmax(eigenvector[0]), 2]*M/eigenvector[1][np.argmax(eigenvector[0]), 0])
    #solutions6.append(x_6)

    x_2 = new_phi_component1/a
    print("pred = ", x_2, "; a = ", a)
    pred.append(x_2)
    x_4 = x_2*b/new_phi_component0
    print("norm_pred = ", x_4)
    norm_pred.append(x_4)
    x_5 = new_phi_component2/a
    solutions5.append(x_5)
    x_6 = x_5*b/new_phi_component0
    solutions6.append(x_6)

    x_3 = x_3 - dt*x_3*x_3*x_3
    print("discrete_sol = ", x_3)
    discrete_sol.append(x_3)

# print the solutions using mathplotlib
t = np.linspace(0, T, nT+1)

# I also want to add the analytically computed solution which is x = 1/np.sqrt(2*(t+0.5))
solutions_analytic = 1/np.sqrt(2*(t+0.5))


# plot the graph of solutions1 in function of t
plt.figure()
# Plot solutions1, pred, discrete_sol, solutions_analytic in the y axis while t is in the x axis

if len(pred) != len(t):
    t = t[:len(pred)]

print(pred)
plt.plot(t, pred, label = "Solution i.e. second element of phi", marker='o')
plt.plot(t, discrete_sol, label = "Discrete Differential Equation solution", marker='o')
plt.plot(t, norm_pred, label = "Solution, normalized by first element of phi", marker='o')
#plt.plot(t, solutions5, label = "Normalization factor, i.e. third element of phi", marker='o')
#plt.plot(t, solutions6, label = "Normalization factor, normalized by first element of phi", marker='o')
#plt.plot(t, solutions_analytic, label = "solutions_analytic")
# add labels to the axes
plt.xlabel("t")
plt.ylabel("x")
# add a title
plt.title("Solution of dx/dt = -x^3 for: x_0 = " + str(x_0) + ", n of copies = " + str(n) + ", M = " + str(M))

# add a legend
plt.legend()
#plt.show()
plt.savefig('diffeq_plot_1.png', dpi=300)
plt.close()


# I want to create a second plot with norm, purity, c_calculated1, c_calculated2, c_analytical, the three values of eigenvector of max eigenvalue.
# the values are already calculated. now we will print them.
plt.figure()
# Plot purities, c1s, c2s, eigenvector_max_0, eigenvector_max_1, eigenvector_max_2 in the y axis while t is in the x axis
plt.plot(t, purities, label = "purity")
plt.plot(t, c1s, label = "c1")
plt.plot(t, c2s, label = "c2")
plt.plot(t, eigenvector_max_0, label = "eigenvector_max_0")
plt.plot(t, eigenvector_max_1, label = "eigenvector_max_1")
plt.plot(t, eigenvector_max_2, label = "eigenvector_max_2")
plt.plot(t, eigenvalue_max, label = "eigenvalue_max")
# add labels to the axes
plt.xlabel("t")
plt.ylabel("values")
# add a title
plt.title("Purity, c1, c2, c_analytical, eigenvector_max_0, eigenvector_max_1, eigenvector_max_2")

# add a legend 
plt.legend()
#plt.show()
plt.savefig('diffeq_plot_2.png', dpi=300)
plt.close()

# NEW PROJECT! I want to define a second order function f(x) = x^2 + k with k.
# Then I want to solve the equation f(x) = 0 for k. Find k.
#def f(x):
#    return x**2 + k

# now we solve the equation f(x) = 0 for k:
#from sympy import Symbol, solve
#k = Symbol('k')
#k = solve(f(x), k)
#print(k)


