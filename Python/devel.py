import sympy as sp
import numpy as np
Delta_t  = 0.01
v = 30.
x = 10
y = 5
theta = 0.5
target = np.array([[3],[1]])
# dt,v_, x_, y_, theta_,g1, g2 = sp.symbols('dt,  v_, x, y, theta, g1, g2')
# vehicle_pos = sp.Matrix([[x_],[y_]])
# rel_pos =  sp.Matrix([[g1],[g2]]) - vehicle_pos
# vehicle_vel = sp.Matrix([[v_*sp.cos(theta_)],[v_*sp.sin(theta_)]])
# norm_relpos = sp.sqrt(sum(sp.matrices.dense.matrix_multiply_elementwise(rel_pos,rel_pos)))
# norm_vehvel = sp.sqrt(sum(sp.matrices.dense.matrix_multiply_elementwise(vehicle_vel,vehicle_vel)))
#
# dot = sum(sp.matrices.dense.matrix_multiply_elementwise(rel_pos,vehicle_vel))
# a = sp.acos(dot/(norm_vehvel*norm_relpos))
# d = sp.diff(a, sp.Matrix([[x_],[y_],[theta_]])).simplify().simplify()
#
# print(d.evalf(subs={dt:Delta_t, v_:v, x_:x , y_:y, theta_:theta,g1: target[0,0], g2:target[1,0]}))
#
# xv, yv, xn, yn = sp.symbols('xv, yv, xn, yn')
# vehicle_pos = sp.Matrix([[xv],[yv]])
# neighbor_pos = sp.Matrix([[xn],[yn]])
# a = vehicle_pos - neighbor_pos
# norm_a = sp.sqrt(sum(sp.matrices.dense.matrix_multiply_elementwise(a,a)))
# print(sp.diff(norm_a, sp.Matrix([[xv],[yv]])).simplify())
# print(sp.diff(norm_a, sp.Matrix([[xn],[yn]])).simplify())


Delta_t, omega, v, x, y, theta = sp.symbols('Delta_t, omega, v, x, y, theta')
mu = v*Delta_t*sp.sinc((omega*Delta_t)/(2*np.pi)) # Division by np.pi to obtain unnormalized sinc i.e., sin(x)/x
x_k1 = x + mu*sp.cos(theta + (omega*Delta_t)/2)
y_k1 = y + mu*sp.sin(theta + (omega*Delta_t)/2)
theta_k1 = theta + omega*Delta_t

state =  sp.Matrix([[x_k1],[y_k1],[theta_k1]])
print(sp.simplify(sp.diff(state,sp.Matrix([[x],[y],[theta]]))))