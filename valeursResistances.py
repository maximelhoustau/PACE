#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 09:59:14 2018

@author: eburle
"""

L1 = [1.1,8.3,67.7,5.8,390,680,120,22,27,270,100,150,7.6,4.8,680,180,220,33,467,148,2.3,473000,470000,21800,8200,75000,180000,68000,564000,218000,81300,39500,6700,56000,33100,4700,75000,9900000,5580000,3300000,1500000,2000000]

L1 = sorted(L1)

n = len(L1)

L2 = []

for i in range(0,n):
   for j in range(i,n):
       r1 = L1[i]
       r2 = L1[j]
       L2.append([r1+r2,r1,"+",r2])
       L2.append([(r1*r2)/(r1+r2),r1,"//",r2])

L2 = sorted(L2)

L3 = []

for i in range(0,n):
   for j in range(0,len(L2)):
       r1 = L1[i]
       r2 = L2[j][0]
       L3.append([r1+r2,r1,"+",L2[j][1:4]])
       L3.append([(r1*r2)/(r1+r2),r1,"//",L2[j][1:4]])

L3 = sorted(L3)

L10 = []
L = []

for i in L1:
    L10.append([i,"valeur initaile"])
    
L = L10 + L2 + L3

L = sorted(L)
N = len(L)

def rechercher1(r):
    a = 0
    b = n - 1
    while (b - a > 1):
        c = (a+b)//2
        if (L1[c] < r):
            a = c
        elif (L1[c] >r):
            b = c
        else:
            a = c
            b = c
    return L1[a]

def trouverPoid1(w, Rf):
    l = rechercher1(Rf/w)
    return l, (Rf/l)

def rechercher3(r):
    a = 0
    b = N - 1
    while (b - a > 1):
        c = (a+b)//2
        if (L[c][0] < r):
            a = c
        elif (L[c][0] >r):
            b = c
        else:
            a = c
            b = c
    return L[a]


def trouverPoid3(w, Rf):
    l = rechercher3(Rf/w)
    return l, (Rf/l[0])

n2 = len(L2)

def rechercher2(r):
    a = 0
    b = n2 - 1
    while (b - a > 1):
        c = (a+b)//2
        if (L2[c][0] < r):
            a = c
        elif (L2[c][0] >r):
            b = c
        else:
            a = c
            b = c
    return L2[a]


def trouverPoid2(w, Rf):
    l = rechercher2(Rf/w)
    return l, (Rf/l[0])

def trouverRf2(liste):
    R = 0
    d = 1000
    for r in L1:
        d1 = 0
        for w in liste:
            d1 += (w-trouverPoid2(w,r)[1])**2
        if (d1 < d):
            d = d1
            R = r
    return R, d

def trouverRf3(liste):
    R = 0
    d = 1000
    for r in L1:
        d1 = 0
        for w in liste:
            d1 += (w-trouverPoid3(w,r)[1])**2
        if (d1 < d):
            d = d1
            R = r
    return R, d

dmax = 0.05

def trouverRes(liste,Rf):
    Liste = []
    m = 0
    dt = 0
    for w in liste:
        v = trouverPoid1(w, Rf)
        d = abs(w-v[1])
        if (d<dmax):
            Liste.append(v)
            m += 1
            dt += d
        else:
            v = trouverPoid2(w, Rf)
            d = abs(w-v[1])
            if (d<dmax):
                Liste.append(v)
                m += 2
                dt += d
            else:
                v = trouverPoid3(w, Rf)
                Liste.append(v)
                m += 3
                dt += abs(w-v[1])
    return Liste, m, dt
    

def toutTrouver(liste,dtmax):
    Rf = 0
    m = 100
    L = []
    for R in L1:
        res = trouverRes(liste,R)
        if ((res[2]<dtmax) and (res[1]<m)):
            m = res[1]
            Rf = R
            L = res
    return Rf, m, L[0]
        
        