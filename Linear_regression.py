import pickle
import numpy as np
import matplotlib.pyplot as plt

fin = open('data.pkl', 'rb')
x = pickle.load(fin)
y = pickle.load(fin)
fin.close()
plt.plot(x, y, 'x')
plt.xlabel('Age')
plt.ylabel('Height')
dx = np.array([np.ones(len(x)), x])
dxT = np.transpose(dx)
pinvdx = np.dot(dxT, np.linalg.inv(np.dot(dx,  dxT)))
W = np.dot( y.T,pinvdx)
r =  np.dot(np.transpose(W), dx)
plt.plot(x, r, '-')
plt.plot(x, y, 'x')
plt.xlabel('Age')
plt.ylabel('Height')
x3 = [1, 3.5]
height3 = np.dot(W.T, x3)
x7 = [1, 7]
height7 = np.dot(W.T, x7)
print("estimated person of age 3.5",height3)
print("estimated person of age 7",height7)
