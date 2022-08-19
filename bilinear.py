import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

def bilinear_map(X, Y, tX, tY):
  srcNum = len(X) * len(X[0])
  print("srcNum:", srcNum)
  targetNum = len(tX) * len(tX[0])
  spacing = X[0,1] - X[0,0]
  print("SPACING:", spacing)

  weighting = np.zeros((targetNum, srcNum))

  t_idx = 0
  for j in range(len(tY)):
    for i in range(len(tX)):
      x = tX[j, i]
      y = tY[j, i]
      #print("x, y:", x, y)
      # Find local square
      xmod = x % spacing
      ymod = y % spacing
      x1 = x - xmod
      y1 = y - ymod
      x2 = x1 + spacing
      y2 = y1 + spacing
      #print("x1, y1, x2, y2:", x1, y1, x2, y2)

      Xinv = np.array([[x2*y2, -y2, -x2, 1],
                       [-x2*y1, y1, x2, -1],
                       [-x1*y2, y2, x1, -1],
                       [x1*y1, -y1, -x1, 1]]) / ((x2 - x1)*(y2 - y1))

      w = np.dot(Xinv, np.array([1, x, y, x * y]))
      #print(w)

      ij_start = np.array([int(round(x1 / spacing)), int(round(y1 / spacing))])
      ijs = np.array([ ij_start + ijd for ijd in np.array([[0,0], [0,1], [1,0], [1,1]]) ])

      #print("ijs:", ijs)
      xindices = np.array([ (ij[1] * len(X[0])) + ij[0] for ij in ijs ])
      #print("XINDICES:", xindices)

      weighting[t_idx, xindices] = w

      t_idx += 1


  return weighting

def get_z(func, X, Y):
  z = []
  for j in range(len(X[0])):
    for i in range(len(X)):
      z.append(func(X[j,i], Y[j,i]))

  return np.array(z, dtype=float)

def twoD(oneD, shape):
  assert shape[0] * shape[1] == len(oneD)
  out = np.zeros(shape)

  ii = 0
  for j in range(shape[1]):
    for i in range(shape[0]):
      out[j][i] = oneD[j * shape[0] + i]
      ii += 1
        
  return out


def sink(x, y):
  return x*x + y*y

def wave(x, y):
  return (np.sin(x) * np.sin(y) + 1)/2

def ripple(x, y):
  rsqrd = x*x + y*y
  return np.exp(-np.sqrt(rsqrd)/2) * (np.cos(rsqrd * 2) + 1) / 2


RESOLUTION = 0.1
space = np.arange(0, 4, RESOLUTION)
print("LEN SPACE:", len(space))
X, Y = np.meshgrid(space, space)

cmap = plt.get_cmap("plasma")

Z = ripple(X, Y)
plt.pcolormesh(X, Y, Z, cmap="plasma", vmin=0.0, vmax=1.0)
plt.gca().set_aspect("equal")
plt.show()

z = get_z(ripple, X, Y)
cols = cmap(z)

print(X)
print(Y)

#plt.plot(X, Y, "x")
for j in range(len(X[0])):
  for i in range(len(X)):
    plt.plot(X[j,i], Y[j,i], "+", c=cols[j * len(X[0]) + i], mew=2, ms=8)


targetspc = np.linspace(min(space)+0.5, max(space)-0.5, 200)
tX, tY = np.meshgrid(targetspc, targetspc)

bmap = bilinear_map(X, Y, tX, tY)
print("--- BMAP: ---")
print(bmap)
print("-------------")

print("Z:",z)
zmapped = np.matmul(bmap, z)
#zmapped = get_z(wave, tX, tY)
print(zmapped)
colsz = cmap(zmapped)

"""
ii = 0
for j in range(len(tX[0])):
  for i in range(len(tX)):
    plt.plot(tX[j,i], tY[j,i], ".", c=colsz[ii], ms=8)
    ii += 1
"""

plt.pcolormesh(tX, tY, twoD(zmapped, (len(tX), len(tX[0]))), cmap="plasma", vmin=0, vmax=1)

plt.gca().set_aspect("equal")
plt.show()

