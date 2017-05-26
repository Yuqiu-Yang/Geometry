

def distance(P1, P2, method = "Homogeneous"):
    import numpy as np
    import math
    # Find the distance between two points
    # P1 and P2 
    P1 = np.asarray(P1)
    P2 = np.asarray(P2)
    if method == "Hyperbolic":
        # check they are on unit hyperboloid
        P1[len(P1) - 1] = -P1[len(P1) - 1]
        return math.acosh(-np.dot(P1,P2))
    elif method == "Spherical":
        # check they are on unit sphere
        return math.acos(np.dot(P1,P2))
    else:
        # check the last coordinate is 1
        return np.dot(P1-P2,P1-P2)

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



