import os, sys
import numpy as np
import sys
sys.path.insert(0, "/Users/ivanzait/Documents/Documents_LM4500/Codes/analysator")
import pytools as pt
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
from data_processing import vdf_tools as vt
#from adaptive_hermite import cubic_transform
from math import comb


BULK_FILE   = "/Users/ivanzait/Documents/Documents_LM4500/Codes/Vlasiator/runs/reconnection_test_small/run_base/bulk.0000000.vlsv"
CELL_COORDS = [0, 0, 0]   # None = center of box; nearest cell with a VDF is used
# ORDER       = 26          # indices l,m,n run 0..ORDER-1 for both transforms
# SP_TH       = 1e-15


def bernstein(k, n, t):
    """k-th Bernstein basis polynomial of degree n on [0, 1]."""
    return comb(n, k) * t**k * (1 - t)**(n - k)
 
 
def bernstein_matrix(n, t):
    """Columns are b_{0,n}(t), ..., b_{n,n}(t). Shape: (len(t), n + 1)."""
    return np.column_stack([bernstein(k, n, t) for k in range(n + 1)])
 
 
def decompose(x, y, n, a=0.0, b=1.0):
    """Least-squares coefficients c_k such that f(x) ~ sum_k c_k b_{k,n}(t),
    with t = (x - a) / (b - a) mapping [a, b] onto [0, 1]."""
    #x = np.linspace(a, b, n_samples)
    t = (x - a) / (b - a)
    B = bernstein_matrix(n, t)
    c, *_ = np.linalg.lstsq(B, y, rcond=None)
    return c
 
 
def reconstruct(c, x, a, b):
    """Evaluate sum_k c_k b_{k,n}(t) at points x."""
    n = len(c) - 1
    t = (x - a) / (b - a)
    return bernstein_matrix(n, t) @ c

def velocity_axis(vlim, vlen, dv): 
    return -vlim + dv / 2.0 + np.arange(vlen) * dv




def main():        
    
    reader = pt.vlsvfile.VlsvReader(BULK_FILE)
    vlim, vlen, dv = vt.get_vdf_parameters(reader)    
    print(vlim, vlen, dv)   
    cellid = vt.get_nearest_vdf_cellid(reader, coords=CELL_COORDS)     
    cube = vt.build_cube(cellid, reader, vlim, vlen, dv)
    f_1d = cube[:, int(vlen/2), int(vlen/2)]
    
    v_axis = velocity_axis(vlim, vlen, dv)
    
    n=30
    c = decompose(v_axis, f_1d, n, -vlim, vlim)    
    
    print('C type:', type(c))
    print('C shape :', c.shape)
    print('C 1 :', c[0])
    
    
    f_rec = reconstruct(c, v_axis, -vlim, vlim)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(v_axis, f_1d, label='original', color='blue')
    ax.plot(v_axis, f_rec, label='reconstructed', color='orange', linestyle='dashed')
    ax.set_xlabel('Velocity')
    ax.set_ylabel('Distribution Function')
    ax.set_title('Bernstein Polynomial Approximation')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    figsave = 'bernstein_approximation.png'
    fig.savefig(figsave, dpi=300)
    
    
    
if __name__ == '__main__':
    main()
