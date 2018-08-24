#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 22:06:52 2018

@author: gsy
"""
import numpy as np

hog =  np.reshape(np.arange(36), [9,4]);

u = np.zeros((4,9));
u0 = u.copy();
u1 = u.copy();
u0[:,0] = 1;
u1[:,1] = 1;
res0 = np.dot(hog, u0);
res1 = np.dot(hog, u1);
