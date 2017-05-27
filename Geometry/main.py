import numpy as np
from prettytable import PrettyTable
###################
# Some useful functions
###################
def genPoint(n, method = "Homogeneous"):
    # Generate random points in R^n, S^(n-1), or L^(n-1)
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
    # Find the distance between two points
    import numpy as np
    import math
    P1 = np.asarray(P1)
    P2 = np.asarray(P2)
    if method == "Hyperbolic":
        eta = np.eye(len(P1))
        eta[-1,-1] = -1
        return math.acosh(abs(np.dot(P1,eta).dot(P2)))
    elif method == "Spherical":
        return math.acos(np.dot(P1,P2))
    else:
        return np.sqrt(np.dot(P1-P2,P1-P2))

def orthoTrans(n, method = "Homogeneous"):
    # Generate elements in O(n) and SO(n) (hyperbolic)
    import numpy as np
    from scipy.stats import ortho_group
    from scipy.stats import special_ortho_group
    if method == "Spherical":
        Q = ortho_group.rvs(n)
        return Q
    elif method == "Hyperbolic":
        # generate pure rotation which 
        # leaves the time coordinate unchanged
        Q = np.zeros((n, n))
        Q[:-1,:-1] = special_ortho_group.rvs(n - 1)
        Q[- 1, - 1] = 1
        return Q
    else:
        # when method is Homogeneous
        # n is the length of the vector - 1
        Q = np.zeros((n + 1, n + 1))
        Q[:-1,:-1] = ortho_group.rvs(n)
        Q[n, n] = 1
        return Q

def translation(n):
    import numpy as np
    U = np.eye(n + 1)
    U[0:-1, -1] = np.random.randn(n)
    return U

def boost(beta):
    # beta 1 by 3 ndarray whose norm is < 1 
    import numpy as np
    Beta = beta.dot(beta.transpose())
    gamma = 1/np.sqrt(1-Beta)
    B = np.zeros((4,4))
    B[3,3] = gamma
    B[3, 0:-1] = -gamma * beta
    B[0:-1, 3] = B[3, 0:-1].transpose()
    B[0:-1,0:-1] = np.matmul(beta.transpose(), beta)
    B[0:-1, 0:-1] *= ((gamma - 1) / Beta) 
    B[0:-1, 0:-1] += np.eye(3)
    return B
  
###################
# Test
###################
def test(P1,P2, trans,method, precision = 3):
    import numpy as np
    import math
    try:
        d = distance(math.inf,math.inf, method = method)
        if not(math.isnan(d)):
            return "Did not pass."
            raise ValueError("distance function seems to return a constant")
    except:
        pass
    d1 = distance(P1, P2, method = method)
    UP1 = trans.dot(P1)
    UP2 = trans.dot(P2)
    d2 = distance(UP1, UP2, method = method)
    d1 = round(d1, precision)
    d2 = round(d2, precision)
    if d1 == d2:
        return "Passed."
    else:
        return "Did not pass."


###################
# Euclidean 
###################
P1 = genPoint(3)
P2 = genPoint(3)
# Orthogonal transformation
O = orthoTrans(3)
# Translation
T = translation(3)
# Translation followed by an Orthogonal transformation
OT = O.dot(T)
# Orthogonal transformation followed by a Translation
TO = T.dot(O)
# 
OP1 = O.dot(P1)
OP2 = O.dot(P2)
TP1 = T.dot(P1)
TP2 = T.dot(P2)
OTP1 = OT.dot(P1)
OTP2 = OT.dot(P2)
TOP1 = TO.dot(P1)
TOP2 = TO.dot(P2)
# Construct tables 
# The table that shows the coordinates of these points
t = PrettyTable(['', 'Coordinate'])
t.add_row(['P1', P1])
t.add_row(['P2', P2])
t.add_row(['OP1', OP1])
t.add_row(['OP2', OP2])
t.add_row(['TP1', TP1])
t.add_row(['TP2', TP2])
t.add_row(['OTP1', OTP1])
t.add_row(['OTP2', OTP2])
t.add_row(['TOP1', TOP1])
t.add_row(['TOP2', TOP2])
print(t)
# The table that shows the distances and if they passed the test
t1 = PrettyTable(['Original', 'Orthogonal', 'Translation',
                  'Ortho + Translation', 'Translation + Ortho'])
t1.add_row([round(distance(P1,P2),3),
            round(distance(OP1,OP2),3),
            round(distance(TP1,TP2),3),
            round(distance(OTP1,OTP2),3),
            round(distance(TOP1,TOP2),3)])
t1.add_row(['',
            test(P1,P2, O,'Homogenous'),
            test(P1,P2, T,'Homogenous'),
            test(P1,P2, OT,'Homogenous'),
            test(P1,P2, TO,'Homogenous')])
print(t1)


###################
# Spherical
###################
P1 = genPoint(4, method = "Spherical")
P2 = genPoint(4, method = "Spherical")
O = orthoTrans(4, method = "Spherical")
O1 = orthoTrans(4, method = "Spherical")

OP1 = O.dot(P1)
OP2 = O.dot(P2)
O1P1 = O1.dot(P1)
O1P2 = O1.dot(P2)

# Construct tables 
# The table that shows the coordinates of these points
t = PrettyTable(['', 'Coordinate'])
t.add_row(['P1', P1])
t.add_row(['P2', P2])
t.add_row(['OP1', OP1])
t.add_row(['OP2', OP2])
t.add_row(['O1P1', O1P1])
t.add_row(['O1P2', O1P2])
print(t)
# The table that shows the distances and if they passed the test
t1 = PrettyTable(['Original', 'Orthogonal', 'Orthogonal1'])
t1.add_row([round(distance(P1,P2, method = "Spherical"),3),
            round(distance(OP1,OP2, method = "Spherical"),3),
            round(distance(O1P1,O1P2, method = "Spherical"),3)])
t1.add_row(['',
            test(P1,P2, O,"Spherical"),
            test(P1,P2, O1,"Spherical")])
print(t1)


###################
# Hyperbolic
###################
P1 = genPoint(4, method = "Hyperbolic")
P2 = genPoint(4, method = "Hyperbolic")
# Pure rotation
O = orthoTrans(4, method = "Hyperbolic")
# Pure boost
beta = np.zeros((1,3))
beta[0,:] = (2*np.random.rand(1)-1) * genPoint(3, method = "Spherical")
T = boost(beta)
# Boost followed by a rotation
OT = O.dot(T)
# Rotation followed by a boost
TO = T.dot(O)
#
OP1 = O.dot(P1)
OP2 = O.dot(P2)
TP1 = T.dot(P1)
TP2 = T.dot(P2)
OTP1 = OT.dot(P1)
OTP2 = OT.dot(P2)
TOP1 = TO.dot(P1)
TOP2 = TO.dot(P2)
# Construct tables 
# The table that shows the coordinates of these points
t = PrettyTable(['', 'Coordinate'])
t.add_row(['P1', P1])
t.add_row(['P2', P2])
t.add_row(['OP1', OP1])
t.add_row(['OP2', OP2])
t.add_row(['TP1', TP1])
t.add_row(['TP2', TP2])
t.add_row(['OTP1', OTP1])
t.add_row(['OTP2', OTP2])
t.add_row(['TOP1', TOP1])
t.add_row(['TOP2', TOP2])
print(t)
# The table that shows the distances and if they passed the test
t1 = PrettyTable(['Original', 'Rotation', 'Boost',
                  'Rotation + Boost', 'Boost + Rotation'])
t1.add_row([round(distance(P1,P2, method = "Hyperbolic"),3),
            round(distance(OP1,OP2, method = "Hyperbolic"),3),
            round(distance(TP1,TP2, method = "Hyperbolic"),3),
            round(distance(OTP1,OTP2, method = "Hyperbolic"),3),
            round(distance(TOP1,TOP2, method = "Hyperbolic"),3)])
t1.add_row(['',
            test(P1,P2, O,"Hyperbolic"),
            test(P1,P2, T,"Hyperbolic"),
            test(P1,P2, OT,"Hyperbolic"),
            test(P1,P2, TO,"Hyperbolic")])
print(t1)
