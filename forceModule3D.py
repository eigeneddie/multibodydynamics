# Copyright (C) 2021 Edgar Sutawika - All Rights Reserve
# For educational purposes

import numpy as np
from calcModule3D import link2index as l2i, uBarSkew as uBSkew

def QeThetaEP(GBarMat, ATrans, uBarSkew, F_R_i):
    uSkew = np.dot(ATrans, uBarSkew)
    matrixA = np.dot(uSkew, GBarMat)
    F_theta_i = -np.transpose(matrixA)*F_R_i

    return F_theta_i

def linSprg3D(GBarMat, uBariP, ATrans, stiffness, riP, rjP, lo):
    Ls = riP-rjP
    LsMag    = np.sqrt(np.transpose(Ls)*Ls) # Scalar
    FsMag    = stiffness*(LsMag-lo) # Scalar
    LsUnit   = Ls/LsMag # unit vector 3x1 
    uBarSkew = uBSkew(uBariP)

    Fs_i     =                         -FsMag*LsUnit # vector 3x1
    Fs_j     =                          FsMag*LsUnit # vector 3x1
    QTheta_i =  FsMag*ATrans@uBarSkew@GBarMat@LsUnit # vector 3x1
    QTheta_j = -FsMag*ATrans@uBarSkew@GBarMat@LsUnit # vector 3x1

    return Fs_i, Fs_j, QTheta_i, QTheta_j

def linDamp3D(GBarMat, uBariP, ATrans, damping, riP_Dot, rjP_Dot):
    LsDot = riP_Dot-rjP_Dot
    LsDotMag    = np.sqrt(np.transpose(LsDot)*LsDot)
    FdMag    = damping*(LsDotMag)
    LsDotUnit   = LsDot/LsDotMag
    uBarSkew = uBSkew(uBariP)

    Fd_i     =         -FdMag*LsDotUnit
    Fd_j     =          FdMag*LsDotUnit
    QTheta_i =  FdMag*ATrans@uBarSkew@GBarMat@LsDotUnit
    QTheta_j = -FdMag*ATrans@uBarSkew@GBarMat@LsDotUnit

    return Fd_i, Fd_j, QTheta_i, QTheta_j


#=========================================


def torSpring(kr, qi, i, j, theta0):
    thetai = qi[l2i(i, "theta")]
    thetaj = qi[l2i(j, "theta")]
    deltaTheta = thetai-thetaj
    Q_SpringThetai = -kr*(deltaTheta-theta0)
    Q_SpringThetaj = kr*(deltaTheta-theta0)

    return Q_SpringThetai, Q_SpringThetaj

def torDamp(cr, qiDot, i, j):
    omegai = qiDot[l2i(i, "theta")]
    omegaj = qiDot[l2i(j, "theta")]
    deltaOmega = omegai-omegaj
    Q_DampThetai = -cr*deltaOmega
    Q_DampThetaj = cr*deltaOmega

    return Q_DampThetai, Q_DampThetaj