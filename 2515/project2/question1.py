import numpy as np
from sklearn.datasets import load_boston


def Huber_loss(a, delta):
    # a=y-t
    # H=np.where(abs(a)<=delta,1/2*a**2,delta*(abs(a)-delta/2))
    dH = np.piecewise(a, [abs(a) <= delta, a > delta, -a < delta], [a, delta, -delta])
    return dH

