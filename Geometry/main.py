import numpy as np
import matplotlib.pyplot as plt

def genPoint(n, method = "Homogeneous"):
    import numpy as np
    P = np.random.randn(n)
    if method == "Hyperbolic":
        P[-1] = np.sqrt(1 + np.dot(P[0 : -1], P[0 : -1]))
        return P
    elif method == "Spherical":
        # Make sure that P is not the origin
        while all(P == 0):
            P = np.random.randn(n)
        P = P / np.sqrt(np.dot(P, P))
        return P
    else:
        tempP = np.zeros(n + 1)
        tempP[0:-1] = P
        tempP[-1] = 1
        return tempP


def distance(P1, P2, method = "Homogeneous"):
    import numpy as np
    import math
    # Find the distance between two points
    P1 = np.asarray(P1)
    P2 = np.asarray(P2)
    if method == "Hyperbolic":
        P1[len(P1) - 1] = -P1[len(P1) - 1]
        return math.acosh(-np.dot(P1,P2))
    elif method == "Spherical":
        return math.acos(np.dot(P1,P2))
    else:
        # check the last coordinate is 1
        return np.sqrt(np.dot(P1-P2,P1-P2))

def orthoTrans(n, method = "Homogeneous"):
    import numpy as np
    from scipy.linalg import qr
    H = np.random.randn(n, n)
    Q, R = qr(H)
    #Q.dot(Q.T)
    #Q.T.dot(Q)
    if method == "Spherical":
        return Q
    else:
        tempQ = np.zeros((n + 1, n + 1))
        tempQ[:-1,:-1] = Q
        tempQ[n, n] = 1
        return tempQ

def translation(n):
    import numpy as np
    U = np.eye(n + 1)
    U[0:-1, -1] = np.random.randn(n)
    return U
    

###################
# Euclidean 
###################

P1 = [3, 4, 5, 1]
P2 = [1, 2, 3, 1]
distance(P1, P2)

# Orthogonal transformation
U = orthoTrans(3)
UP1 = U.dot(P1)
UP2 = U.dot(P2)
distance(UP1, UP2)

# Translation 
U = translation(3)
UP1 = U.dot(P1)
UP2 = U.dot(P2)
distance(UP1, UP2)

###################
# Spherical
###################
P1 = np.asarray([3, 4, 5, 1])
P2 = np.asarray([1, 2, 3, 1])

P1 = P1 / np.sqrt(P1.dot(P1))
P2 = P2 / np.sqrt(P2.dot(P2))

distance(P1, P2, method = "Spherical")

U = orthoTrans(4, method = "Spherical")
UP1 = U.dot(P1)
UP2 = U.dot(P2)
distance(UP1, UP2, method = "Spherical")

###################
# Hyperbolic
###################







       

