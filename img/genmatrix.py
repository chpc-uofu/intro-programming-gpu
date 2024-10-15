import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

SZ=8
A=np.zeros([SZ,SZ])
A[2,4]=1

plt.matshow(A,cmap=ListedColormap(['green', 'gold']), extent=(0, SZ, SZ, 0))
plt.xticks(range(0,SZ))
plt.yticks(range(0,SZ))
plt.xlabel(r"$\mathbf{j}$", loc='center')
plt.ylabel(r"$\mathbf{i}$", rotation=0.0)
plt.tick_params(axis='x', bottom=False)
plt.arrow(2,2.5,dx=2,dy=0, head_width=0.20, head_length=0.4,color='blue',length_includes_head=False)
plt.text(x=0.5,y=2.5,s="P[i=2,j=4]")

plt.grid(c='red', ls=':', lw='1.0')
plt.show()

