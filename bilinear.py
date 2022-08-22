import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.sparse import csc_matrix

def bilinear_map(X, Y, tX, tY):
  srcNum = len(X) * len(X[0])
  print("srcNum:", srcNum)
  targetNum = len(tX) * len(tX[0])
  spacingx = X[0,1] - X[0,0]
  spacingy = Y[1,0] - Y[0,0]
  print("SPACING:", spacingx, spacingy)

  map_shape = (targetNum, srcNum)

  row = []
  col = []
  ws = []

  t_idx = 0
  for j in range(len(tY)):
    for i in range(len(tX)):
      x = tX[j, i]
      y = tY[j, i]
      #print("x, y:", x, y)
      # Find local square
      xmod = x % spacingx
      ymod = y % spacingy
      x1 = x - xmod
      y1 = y - ymod
      x2 = x1 + spacingx
      y2 = y1 + spacingy
      #print("x1, y1, x2, y2:", x1, y1, x2, y2)

      Xinv = np.array([[x2*y2, -y2, -x2, 1],
                       [-x2*y1, y1, x2, -1],
                       [-x1*y2, y2, x1, -1],
                       [x1*y1, -y1, -x1, 1]]) / ((x2 - x1)*(y2 - y1))

      w = np.dot(Xinv, np.array([1, x, y, x * y]))
      #print(w)

      ij_start = np.array([int(round(x1 / spacingx)), int(round(y1 / spacingy))])
      ijs = np.array([ ij_start + ijd for ijd in np.array([[0,0], [0,1], [1,0], [1,1]]) ])

      #print("ijs:", ijs)
      #xindices =       #print("XINDICES:", xindices)

      for ii, xx in enumerate(np.array([ (ij[1] * len(X[0])) + ij[0] for ij in ijs ])):
        row.append(t_idx)
        col.append(xx)
        #weighting[t_idx, xx] = w[ii]
        ws.append(w[ii])

      t_idx += 1


  ws = np.array(ws)
  row = np.array(row)
  col = np.array(col)
  
  return csc_matrix((ws, (row, col)), shape=map_shape)


def compare_with_control(x1d, y1d, zs, tX1d, tY1d, tX, tY, tZ):
  cont = interp2d(x1d, y1d, zs)
  Zcontrol = cont(tX1d, tY1d)

  # NOTE: Scipy treats function other way around; i, j not j, i
  plt.pcolormesh(tY, tX, Zcontrol, cmap="plasma", vmin=0, vmax=1, zorder=10)
  plt.title("CONTROL")
  plt.gca().set_aspect("equal")
  plt.show()

  tZ2d = twoD(zmapped, (len(tX), len(tX[0])))
  diff = Zcontrol - tZ2d.T

  plt.pcolormesh(tX, tY, diff, cmap="plasma", zorder=10)
  plt.colorbar()
  plt.title("$\Delta$")

  plt.gca().set_aspect("equal")
  plt.show()

def compare_with_real(func, tX1d, tY1d, tX, tY, tZ):
  Zcontrol = func(tX, tY)

  plt.pcolormesh(tX, tY, Zcontrol, cmap="plasma", vmin=0, vmax=1, zorder=10)
  plt.title("REAL")
  plt.gca().set_aspect("equal")
  plt.show()

  tZ2d = twoD(zmapped, (len(tX), len(tX[0])))
  diff = Zcontrol - tZ2d

  plt.pcolormesh(tX, tY, diff, cmap="plasma", zorder=10)
  plt.colorbar()
  plt.title("$\Delta$real")

  plt.gca().set_aspect("equal")
  plt.show()


def get_z(func, X, Y):
  z = []
  for j in range(len(X[0])):
    for i in range(len(X)):
      z.append(func(X[j,i], Y[j,i]))

  return np.array(z, dtype=float)

def oneD(twoD):
  assert len(twoD.shape) == 2

  out = np.zeros(twoD.shape[0] * twoD.shape[1])
  for j in range(twoD.shape[1]):
    for i in range(twoD.shape[0]):
      out[j * twoD.shape[0] + i] = twoD[j][i]
  
  return out


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
  xtmp = x - 2
  ytmp = y - 2
  v = abs(xtmp * 2) * abs(ytmp * 2)
  return v # np.sqrt(xtmp*xtmp + ytmp*ytmp)

def wave(x, y):
  Wfac = 2
  return (np.sin(y * x * x * Wfac) * np.sin(y * Wfac) + 1)/2

def waveplus(x, y):
  Wfac = 6
  return (np.sin(x*y * Wfac) * np.sin(y * Wfac) + 1)/2

def ripple(x, y):
  rsqrd = x*x + y*y
  return np.exp(-np.sqrt(rsqrd)/2) * (np.cos(rsqrd * 2) + 1) / 2

def square_ripple(x, y):
  SER = 3
  rsqrd = x*x + y*y
  r = np.sqrt(rsqrd)**3.14159
  decay = 1 # np.exp(-np.sqrt(rsqrd)/2)
  
  z = 0
  for k in range(1, SER+1):
    kk = 2 * k - 1
    z += (1/kk) * np.sin(2 * np.pi * kk * r)

  z *= 4.0/np.pi
  z = (z + 1)/2

  return decay * z

def checkers(x, y):
  SER = 10
  decay = 1 # np.exp(-np.sqrt(rsqrd)/2)
  
  z = 0
  for k in range(1, SER+1):
    kk = 2 * k - 1
    z += (1/kk) * np.sin(2 * np.pi * kk * x) * np.sin(2 * np.pi * kk * y)

  z *= 4.0/np.pi
  z = (z + 1)/2

  return decay * z

RESOLUTION = 0.05
x1d = y1d = np.arange(0, 4, RESOLUTION)
X, Y = np.meshgrid(x1d, y1d)

cmap = plt.get_cmap("plasma")

F = checkers
Z = F(X, Y)
plt.pcolormesh(X, Y, Z, cmap="plasma", vmin=0.0, vmax=1.0)
plt.gca().set_aspect("equal")
plt.show()

z = get_z(F, X, Y)
cols = cmap(z)

print(X)
print(Y)

#plt.plot(X, Y, "x")
for j in range(len(X[0])):
  for i in range(len(X)):
    plt.plot(X[j,i], Y[j,i], "+", c=cols[j * len(X[0]) + i], mew=2, ms=8)


tx1d = ty1d = np.linspace(min(x1d)+0.5, max(x1d)-0.5, 300)

tX, tY = np.meshgrid(tx1d, ty1d)

bmap = bilinear_map(X, Y, tX, tY)
print("--- BMAP: ---")
print(bmap)
print("-------------")

print("Z:",z)
zmapped = bmap * z
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

plt.pcolormesh(tX, tY, twoD(zmapped, (len(tX), len(tX[0]))), cmap="plasma", vmin=0, vmax=1, zorder=10)

plt.gca().set_aspect("equal")
plt.show()

#(x1d, y1d, zs, tX1d, tY1d, tX, tY, tZ)
compare_with_control(x1d, y1d, z, tx1d, ty1d, tX, tY, zmapped)
#(func, tX1d, tY1d, tX, tY, tZ)
compare_with_real(F, tx1d, ty1d, tX, tY, zmapped)
