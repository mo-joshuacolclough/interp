import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.sparse import dia_matrix

CLRMAP = "viridis"

@np.vectorize
def sink(x, y):
  x -= 3
  y -= 3
  return pow(x, 3) + pow(y, 3)

@np.vectorize
def wavy(x, y):
  #return (np.sin(x) * np.cos(y) * np.sin(x*y) + 1)/2
  return (np.sin(4*x) * np.cos(y) + 1)/2

@np.vectorize
def ripple(x, y):
  x -= 4.5
  y -= 4.5
  rsqrd = x*x + y*y
  r = np.sqrt(rsqrd)

  return np.exp(-r/4) * ((np.cos(rsqrd * 2) + 1) / 2)

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

# Z needs to be array
def bilinear_map(xv, yv, target_points):
  weighting = np.zeros((len(target_points), len(xv) * len(yv)))

  spacing = xv[1,0] - xv[0,0]

  t_idx = 0
  for t in target_points:
    # Find Q11 .. Q22
    xmod = t[0] % spacing
    ymod = t[1] % spacing

    x1 = t[0] - xmod
    x2 = x1 + spacing
    y1 = t[1] - ymod
    y2 = y1 + spacing
    
    Xinv = np.array([[x2*y2, -y2, -x2, 1],
                     [-x2*y1, y1, x2, -1],
                     [-x1*y2, y2, x1, -1],
                     [x1*y1, -y1, -x1, 1]]) / ((x2 - x1)*(y2 - y1))

    w = np.dot(Xinv, np.array([1, t[0], t[1], t[0] * t[1]]))

    # Bottom left of current square
    ij_start = np.array([int((x1 - xv[0,0]) / spacing), int((y1 - yv[0,0]) / spacing)])

    for idx, ij in enumerate(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])):
      ij_curr = ij_start + i
      weighting[t_idx, ij_curr[1]*len(xv) + ij_curr[0]] = w[idx]
      
    t_idx += 1

  return weighting

def zvals_for_tile(zs, x1, y1, x2, y2, startx, starty):
  spacing = x2 - x1
  ij_start = np.array([int((x1 - startx) // spacing), int((y1 - starty) // spacing)])
  #print("IJ START:", ij_start)
  return np.array([zs[ij_start[0] + ij[0], ij_start[1] + ij[1]] for ij in [np.array([0, 0]), np.array([0, 1]), np.array([1,0]), np.array([1, 1])]])


def bilinear(x, y, xv, yv, zs):

  def get_x1y1x2y2(x, y, spacing):
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
    #print("IJ:\t", ij)
    fs.append(zs[ij[0], ij[1]])

  fs = np.array(fs)

  Xinv = np.array([[x2*y2, -y2, -x2, 1],
                   [-x2*y1, y1, x2, -1],
                   [-x1*y2, y2, x1, -1],
                   [x1*y1, -y1, -x1, 1]]) / ((x2 - x1)*(y2 - y1))

  weighting = np.dot(Xinv, np.array([1, x, y, x * y]))

  return np.dot(weighting, fs)

def bilinear_rand(N, bounds, ax):
  xbound = np.array(bounds[0])
  ybound = np.array(bounds[1])
  size = np.array([xbound[1] - xbound[0], ybound[1] - ybound[0]])
  samples = np.random.rand(N, 2) * size + np.array([xbound[0], ybound[0]])

  zsnew = np.zeros(N)

  for i in range(len(samples)):
    z = bilinear(samples[i][0], samples[i][1], xv, yv, zs)
    # col = plasma(z)
    zsnew[i] = z
    ax.plot(samples[i][0], samples[i][1], ".", c=(0.0, 0.0, 0.0, 0.2), markersize=2)
    #ax.plot(samples[i][0], samples[i][1], ".", c=col, markersize=8)
 
  return samples, zsnew

def bilinear_samp(xv, yv, zs, xv_sample, yv_sample, sample_sizex, sample_sizey, ax):
  zsnew = np.zeros((sample_sizex, sample_sizey))
  for i in range(sample_sizex):
    for j in range(sample_sizey):
      b = bilinear(xv_sample[i,j], yv_sample[i,j], xv, yv, zs)
      col = np.array(plasma(b))
      #plt.plot(xv_sample[i,j], yv_sample[i,j], ".", c=(0.0, 0.0, 0.0, 0.2), markersize=2)
      ax.plot(xv_sample[i,j], yv_sample[i,j], ".", c=col, markersize=8)
      zsnew[i,j] = b

  return zsnew


FUNC = wavy
#RESOLUTION = 0.338
#RESOLUTION = 1.326783
#RESOLUTION = 3.3
#RESOLUTION = 0.04888
RESOLUTION = 0.99
xrng = (-1, 4)
yrng = (-1, 4)

x = np.arange(xrng[0], xrng[1], RESOLUTION)
y = np.arange(yrng[0], yrng[1], RESOLUTION)
xv, yv = np.meshgrid(x, y, indexing="ij")
Nx = len(xv)
Ny = len(yv[0])

print("Nx: %d, Ny: %d" % (Nx, Ny))
print("EXPECTED NUMBER OF SAMPLES:\t", Nx * Ny)
vals = []

plasma = plt.get_cmap(CLRMAP)
#grscl = plt.get_cmap("viridis")

fig, ax = plt.subplots()
zs = np.zeros((Nx, Ny))
for i in range(Nx):
  for j in range(Ny):
    z = FUNC(xv[i,j], yv[i,j])
    col = plasma(z)
    ax.plot(xv[i,j], yv[i,j], "+", c=col, zorder=5, markersize=8, mew=2)

    #ax.plot(xv[i,j], yv[i,j], "+k", zorder=5) 
    #plt.plot(xv[i,j], yv[i,j], ".", c=col, zorder=1, markersize=4)
    zs[i,j] = z

def render():
  i = 0
  for N in range(2, 4):
    print("Doing frame %d: N = %d." % (i, N))
    fig, ax = plt.subplots()

    xtmp = np.linspace(xrng[0], xrng[1], num=N)
    ytmp = np.linspace(yrng[0], yrng[1], num=N)
    xv_sample, yv_sample = np.meshgrid(xtmp, ytmp, indexing="ij")
    #control(xv, yv, zs, xv_sample, yv_sample, len(xtmp))
    bilinear_samp(xv, yv, zs, xv_sample, yv_sample, len(xtmp), len(ytmp), ax)
    ax.set_title("Bilinear: $N = %d^2$\n$f(x,y) = \sin(x) \cos(y) \sin(x*y)$" % N)

    #bilinear_rand(N, [xrng, yrng], ax)
    #ax.set_title("Bilinear random: $N = %d^2$\n$f(x,y) = \sin(4x)\cos(y)$" % n)

    plt.gca().set_aspect("equal")
    ax.set_xlim(xrng)
    ax.set_ylim(yrng)

    plt.savefig("/net/spice/scratch/jcolclou/interpolate/frames/frame%d.png" % i, dpi=300)
    plt.close(fig)
    i += 1

def meshgrid_to_points(xv, yv):
  points = np.zeros((len(xv) * len(yv), 2))
  for i in range(len(xv)):
    for j in range(len(yv)):
      points[j * len(xv) + i] = np.array([xv[i,j], yv[i,j]])

  return points

def main():
  S = False
  
  N = 100
  Nsqrt = int(np.sqrt(N))
  print("xrng:", xrng, "yrng:", yrng)
  xtmp = np.linspace(xrng[0]+0.25, xrng[1]-0.25, num=Nsqrt)
  ytmp = np.linspace(yrng[0]+0.25, yrng[1]-0.25, num=Nsqrt)
  print(xtmp)
  xv_sample, yv_sample = np.meshgrid(xtmp, ytmp, indexing="ij")

  # Back to points
  samples = meshgrid_to_points(xv_sample, yv_sample)
  # NOTE: DEBUG
  # S = True
  #samples = np.array([[8.22, 6.22], [4.66, 5.86]])
  #samples = np.random.rand(100, 2) * 6.0 + np.array([2.0, 2.0])

  zpoints = zs.flatten() # meshgrid_zs_to_points(zs, (len(xv), len(yv)))

  #control(xv, yv, zs, xv_sample, yv_sample, len(xtmp))
  #z_sample = bilinear_samp(xv, yv, zs, xv_sample, yv_sample, len(xtmp), len(ytmp), ax)
  #ax.contourf(xv_sample, yv_sample, z_sample, levels=256, cmap=CLRMAP)

  #ax.set_title("Bilinear random: $N = %d^2$\n$f(x,y) = \sin(4x)\cos(y)$" % N)

  #samples, z_sample = bilinear_rand(N, [xrng, yrng], ax)
  #ax.tricontourf(samples[:,0], samples[:,1], zsnew, levels=256, cmap=CLRMAP)
  #ax.set_title("Bilinear random: $N = %d^2$\n$f(x,y) = \sin(4x)\cos(y)$" % n)

  # """ MAPPING
  bmap = bilinear_map(xv, yv, samples)
  print("bmap shape:", bmap.shape, ", z shape:", zpoints.shape)
  mapped = np.matmul(bmap, zpoints)
  print("Mapped?:", mapped)
  print("Map shaped:", mapped.shape)

  mapped_mgrd = np.reshape(mapped, (len(xv_sample),len(yv_sample)))
  print("Mapped mgrd", mapped_mgrd)

  cols = plasma(mapped)
  if S:
    for i, c in enumerate(cols):
      ax.plot(samples[i,0], samples[i,1], ".", c=c, ms=6)
  else:
    for i in range(len(xv_sample)):
      for j in range(len(yv_sample)):
        col = plasma(mapped[j * len(xv_sample) + i])
        ax.plot(xv_sample[i, j], yv_sample[i, j], ".", c=col, ms=6)

  #ax.pcolormesh(xv_sample, yv_sample, mapped_mgrd)
  #ax.pcolormesh(xv, yv, zs)

  # """

  # TEST AGAINST SCIPY
  #f_control = interp2d(xv, yv, zs, kind="cubic")
  #z_control = f_control(xv_sample, yv_sample)

  """
  # Find error
  z_true = FUNC(xv_sample, yv_sample)
  #z_true = FUNC(samples[:,0], samples[:,1])
  err = (z_true - z_sample)/z_true
  levels = [0.1 * i for i in range(-int(1/0.1), int(1/0.1))]
  cm = ax.contourf(xv_sample, yv_sample, err, cmap="seismic", levels=levels)
  #cm = ax.tricontourf(samples[:,0], samples[:,1], err, levels=[0.1 * i for i in range(int(1/0.1))], cmap="plasma")
  plt.colorbar(cm)
  """

  plt.gca().set_aspect("equal")
  ax.set_xlim(xrng)
  ax.set_ylim(yrng)
  plt.show()

main()

