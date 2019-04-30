import numpy as NP
import matplotlib.pyplot as PLT

def build_checkerboard(w, h) :
      re = NP.r_[ w*[0,1] ]              # even-numbered rows
      ro = NP.r_[ w*[1,0] ]              # odd-numbered rows
      return NP.row_stack(h*(re, ro))


checkerboard = build_checkerboard(5, 5)

fig, ax = PLT.subplots()
ax.imshow(checkerboard, cmap=PLT.cm.gray, interpolation='nearest')
PLT.show()


