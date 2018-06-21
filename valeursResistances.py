#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 09:59:14 2018

@author: eburle
"""

L1 = [1.1,8.3,380,67.7,5.8,380,680,120,22,27,270,100,150,7.6,4.8,680,180,220,33,467,148,2.3,473000,470000,21800,8200,75000,180000,68000,564000,218000,81300,39500,6700,56000,33100,4700,75000,9900000,5580000,3300000,1500000,2000000]

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
    L10.append([i,1])
    L.append(i)

L20 = []

for i in L2:
    L20.append([i[0],2])
    L.append(i[0])
    
L30 = []

for i in L3:
    L30.append([i[0],3])
    L.append(i[0])
    
Lf = L10 + L20 + L30

Lf = sorted(Lf)

L = sorted(L)