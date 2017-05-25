
def distance(P1, P2, method = "Euclidean"):
    # Find the distance between two points
    # P1 and P2 
    import numpy as np
    import math
    P1 = np.asarray(P1)
    P2 = np.asarray(P2)
    if method == "Hyperbolic":
        # check they are on unit hyperboloid
        P1[len(P1) - 1] = -P1[len(P1) - 1]
        return -math.acosh(np.dot(P1,P2))
    elif method == "Spherical":
        # check they are on unit sphere
        return math.acos(np.dot(P1,P2))
    else:
        # check the last coordinate is 1
        return np.dot(P1-P2,P1-P2)



