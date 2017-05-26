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
        # when method is Homogeneous
        # n is the length of the desired 
        # vector - 1 
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
        return np.sqrt(np.dot(P1-P2,P1-P2))

def orthoTrans(n, method = "Homogeneous"):
    import numpy as np
    from scipy.stats import ortho_group
    from scipy.stats import special_ortho_group
    if method == "Spherical":
        Q = ortho_group.rvs(n)
        return Q
    elif method == "Hyperbolic":
        # generate pure rotation which 
        # leaves the time coordinate unchanged
        Q = special_ortho_group.rvs(n - 1)
        tempQ = np.zeros((n + 1, n + 1))
        tempQ[:-1,:-1] = Q
        tempQ[n, n] = 1
        return tempQ
    else:
        # when method is Homogeneous
        # n is the length of the vector - 1
        Q = ortho_group.rvs(n)
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
P1 = genPoint(3)
P2 = genPoint(3)
print(P1)
print(P2)
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
P1 = genPoint(4, method = "Spherical")
P2 = genPoint(4, method = "Spherical")


distance(P1, P2, method = "Spherical")

U = orthoTrans(4, method = "Spherical")
UP1 = U.dot(P1)
UP2 = U.dot(P2)
distance(UP1, UP2, method = "Spherical")

###################
# Hyperbolic
###################
P1 = genPoint(4, method = "Hyperbolic")
P2 = genPoint(4, method = "Hyperbolic")
distance(P1, P2, method = "Hyperbolic")










