import numpy as np
import mdtraj as md

def _best_hummer_q(trajectory, signature, r0):
    r = md.compute_distances(trajectory, signature)
    q = np.mean(1.0 / (1 + np.exp(50. * (r - 1.8 * r0))), axis=1)
    return q

signature = np.load('signature.npy')
r0 = np.load('r0.npy')

def inB(trajectory):
    return _best_hummer_q(trajectory, signature, r0) <= .01
