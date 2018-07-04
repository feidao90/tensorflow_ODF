import tensorflow as  tf
import numpy as np

import matplotlib.pyplot as plt

x_np = np.random.rand(2,2)

identify_matrix = tf.diag([1.,1.,1.])
identify_matrix_two = tf.diag([2.,2.,2.])
print(sess.run(identify_matrix)) #从一维数组，生成对角矩阵
print(sess.run(identify_matrix_two))

A = tf.truncated_normal([2,3])
print("A:")
print(sess.run(A))

B = tf.fill([2,3],5.0)
print("B:")
print(sess.run(B))

C = tf.random_uniform([3,2])
print("C:")
print(sess.run(C))

D = tf.convert_to_tensor(np.array([[1.,2.,3.],[-3.,-7.,-1.],[0.,5.,-2.]]))
print("D:")
print(sess.run(D))

#make graphise

x_initial = tf.constant(1.0)
y_initial = tf.constant(1.0)
x_t1 = tf.Variable(x_initial)
y_t1 = tf.Variable(y_initial)

# make the placeholders
t_delta = tf.placeholder(tf.float32,shape=())
a = tf.placeholder(tf.float32,shape=())
b = tf.placeholder(tf.float32,shape=())
c = tf.placeholder(tf.float32,shape=())
d = tf.placeholder(tf.float32,shape=())
x_t2 = x_t1 + (a*x_t1 + b*x_t1*y_t1)*t_delta
y_t2 = y_t1 + (c*y_t1 + d*x_t1*y_t1)*t_delta

step = tf.group(x_t1.assign(x_t2),y_t1.assign(y_t2))

init = tf.global_variables_initializer()
sess.run(init)

#Run the ODE
prey_values = []
predator_values = []

for i in  range(1000):
    step.run({a : (2./3.), b : (-4./3.), c : -1.0, d : 1.0, t_delta : 0.01},session = sess)
    temp_prey,temp_pred = sess.run([x_t1,y_t1])
    prey_values.append(temp_prey)
    predator_values.append(temp_pred)

# draw
plt.plot(prey_values,label = "prey")
plt.plot(predator_values,label = "predator")
plt.legend(loc = 'upper right')
plt.show()
print("end")
