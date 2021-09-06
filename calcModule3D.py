# Copyright (C) 2021 Edgar Sutawika - All Rights Reserve
# For educational purposes

# frequently used calculation functions
import numpy as np
import pandas as pd

def ATrans (theta0, theta1, theta2, theta3): #A
    
    a11 = 1-2*theta2**2-2*theta3**2
    a12 = 2*(theta1*theta2-theta0*theta3)
    a13 = 2*(theta1*theta3+theta0*theta2)

    a21 = 2*(theta1*theta2+theta0*theta3)
    a22 = 1-2*theta1**2-2*theta3**2
    a23 = 2*(theta3*theta2-theta0*theta1)

    a31 = 2*(theta1*theta3-theta0*theta2)
    a32 = 2*(theta3*theta2+theta0*theta1)
    a33 = 1-2*theta1**2-2*theta2**2

    Atrans = np.array([[a11, a12, a13],
                        [a21, a22, a23],
                        [a31, a32, a33]])
    return Atrans

def uBarSkew(ubiP):

    ubarSkew = np.array([[      0, -ubiP[2],  ubiP[1]],
                         [ubiP[2],        0, -ubiP[0]],
                         [-ubiP[1], ubiP[0],        0]])

    return ubarSkew

def GBarMat(theta0, theta1, theta2, theta3):
    GbarMat = np.array([[-theta1, theta0, theta3, -theta2],
                        [-theta2, -theta3, theta0, theta1],
                        [-theta3, theta2, -theta1, theta0]])

    return 2*GbarMat


def local2global3D(qi, u_bar_iP, link_i):
    # To calculate Point of Interest positions in terms of global coordinates
    index_i = link2index(link_i, "x")
    id_Theta = link2index(link_i, "t0")

    Ri = np.array([qi[index_i], qi[index_i+1], qi[index_i+2]]) 
    t0, t1, t2, t3 = qi[id_Theta], qi[id_Theta+1], qi[id_Theta+2], qi[id_Theta+3]

    A_matrix = ATrans(t0, t1, t2, t3)

    riP = Ri + A_matrix@u_bar_iP
    
    return riP

def local2globalDot3D(qi, qiDot, u_bar_iP, link_i):
    # To calculate Point of Interest positions in terms of global coordinates
    id_R = link2index(link_i, "x")
    id_Theta = link2index(link_i, "t0")
    
    Ri_Dot = np.array([qiDot[id_R], qiDot[id_R+1], qiDot[id_R+2]]) 
    t0, t1, t2, t3 = qi[id_Theta], qi[id_Theta+1], qi[id_Theta+2], qi[id_Theta+3]
    
    A_matrix = ATrans(t0, t1, t2, t3)
    ubarSkewMat = uBarSkew(u_bar_iP)
    GBar = GBarMat(t0, t1, t2, t3)
    thetaDot = np.array([ [qiDot[id_R], qiDot[id_R+1], qiDot[id_R+2], qiDot[id_R+3]] ])

    riP_Dot = Ri_Dot - A_matrix @ ubarSkewMat @ GBar @ thetaDot

    return riP_Dot

def link2index(link, string):
    if string == "x":
        index = 7*(link-1)
    elif string == "y":
        index = 7*(link-1)+1
    elif string == "z":
        index = 7*(link-1)+2
    elif string == "t0":
        index = 7*(link-1)+3
    elif string == "t1":
        index = 7*(link-1)+4
    elif string == "t2":
        index = 7*(link-1)+5
    elif string == "t3":
        index = 7*(link-1)+6

    index = int(index)
    return index