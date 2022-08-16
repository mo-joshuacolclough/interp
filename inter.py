import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

@np.vectorize
def sink(x, y):
  return pow(x, 2) + pow(y, 2)

@np.vectorize
def wavy(x, y):
  return (np.sin(3*x) * np.cos(y) + 1)/2

# interp
def control(xv, yv, zs, xv_sample, yv_sample, size):
  interpfunc = interpolate.interp2d(xv, yv, zs)
  for i in range(size):
    for j in range(size):
      zsampl = interpfunc(xv_sample[i,j], yv_sample[i,j])
      col = plasma(zsampl)
      plt.plot(xv_sample[i,j], yv_sample[i,j], ".", c=col, zorder=0)
 
  # -- RAND --
  #Nsampl = 1000
  #samples = np.random.rand(Nsampl,2) * 10.0
  #for i in range(len(samples)):
  #  zsampl = interpfunc(samples[i][0], samples[i][1])
  #  col = plasma(zsampl)
  #  plt.plot(samples[i][0], samples[i][1], ".", c=col, zorder=0)

# control(xv, yv, zs)

def bilinear(x, y, xv, yv, zs):

  def get_x1y1x2y2(x, y, spacing):
    #xmod = np.floor(x/spacing) * spacing
    #ymod = np.floor(y/spacing) * spacing
    
    #xmod = (abs(x) % spacing) * np.sign(x)
    #ymod = (abs(y) % spacing) * np.sign(y)

    xmod = x % spacing
    ymod = y % spacing

    x1 = x - xmod
    x2 = x1 + spacing
    y1 = y - ymod
    y2 = y1 + spacing
    return x1, x2, y1, y2

  Nx = len(xv)
  Ny = len(yv)
  spacing = xv[1,0] - xv[0,0]
  startx = xv[0,0]
  starty = yv[0,0]
  endx = xv[-1,0]
  datasize = endx - startx
  #print("STARTX, STARTY:\t", startx, starty)
  #print("SPACING:", spacing)
  x1, x2, y1, y2 = get_x1y1x2y2(x, y, spacing)
  #print("x1,x2,y1,y2:\t", x1, x2, y1, y2)
  ij_start = np.array([int((x - startx) // spacing), int((y - starty) // spacing)])
  #print("ij_start:\t", ij_start)
  fs = []  # values at nearest locations
  f = 0
  for ij_d in np.array([[0,0], [0,1], [1,0], [1,1]]):
    ij = ij_start + ij_d
    print("IJ:\t", ij)
    fs.append(zs[min(ij[0], Nx-1), min(ij[1], Ny-1)])

  fs = np.array(fs)

  X = np.array([[    1,     1,     1,     1],
                [   x1,    x1,    x2,    x2],
                [   y1,    y2,    y1,    y2],
                [x1*y1, x1*y2, x2*y1, x2*y2]])

  Xinv = np.linalg.inv(X)
  weighting = np.dot(Xinv, np.array([1, x, y, x * y]))
  # print("Weighting:", weighting)

  return np.dot(weighting, fs)

def bilinear_rand(N, size, focus=np.zeros(2)):
  samples = (np.random.rand(N, 2) * size) - size*0.5 + focus
  for i in range(len(samples)):
    zsampl = bilinear(samples[i][0], samples[i][1], xv, yv, zs)
    col = plasma(zsampl)
    plt.plot(samples[i][0], samples[i][1], ".", c=col, markersize=8)

def bilinear_samp(xv, yv, zs, xv_sample, yv_sample, sample_sizex, sample_sizey):
  for i in range(sample_sizex):
    for j in range(sample_sizey):
      b = bilinear(xv_sample[i,j], yv_sample[i,j], xv, yv, zs)
      col = plasma(b)
      plt.plot(xv_sample[i,j], yv_sample[i,j], ".", c=col, markersize=8)

FUNC = wavy
RESOLUTION = 0.338
x = np.arange(0, 6, RESOLUTION)
y = np.arange(0, 6, RESOLUTION)
N = len(x)
vals = []
zs = np.zeros((len(x), len(y)))

plasma = plt.get_cmap("plasma")
grscl = plt.get_cmap("viridis")

xv, yv = np.meshgrid(x, y, indexing="ij")

for i in range(N):
  for j in range(N):
    z = FUNC(xv[i,j], yv[i,j])
    col = plasma(z)
    plt.plot(xv[i,j], yv[i,j], "+", c=col, zorder=1, markersize=10, mew=3)
    #plt.plot(xv[i,j], yv[i,j], ".", c=col, zorder=1, markersize=4)
    zs[i,j] = z

xtmp = np.arange(2, 4, 0.1)
ytmp = np.arange(1, 3, 0.1)
xv_sample, yv_sample = np.meshgrid(xtmp, ytmp, indexing="ij")
#control(xv, yv, zs, xv_sample, yv_sample, len(xtmp))
bilinear_samp(xv, yv, zs, xv_sample, yv_sample, len(xtmp), len(ytmp))
#bilinear_rand(int(1e4), np.array([3, 2]))

plt.gca().set_aspect("equal")
plt.show()

