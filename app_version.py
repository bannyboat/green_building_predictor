# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 2021

@author: iuliia.glushakova@polyu.edu.hk

This script is an example of how to use the FC and AD framework.
It allows to call pre-trained models and by feeding
the input data get predictions of cost or/and duration.

Note:   this algorithm works for one project. 
        If you want to predict several projects 
        at once, you'll have to modify it.
"""

# Importing libraries
import numpy as np
import framework
from framework import *  # Framework's library. 


def main():

    # Suppose a user picked prediction of cost, which is FC (final cost) or AD (actual duration)
    # I make it a variable, so you can switch it easily
    problem = "FC"    
    
    # X consists of 16 input features, the features should follow the same order    
    # as in the training stage
    PT = 3 
    AS = 8000.0 # 
    AT = 1
    OB = 135.94
    PD = 1.74
    SM = 8
    SY = 2012
    SS = 76.0
    MW = 50.0
    EU = 68.0
    WU = 71.0
    HWB = 82.0
    IA = 5

    # Number of cold, hot and rainy days is calculated according to HK Observatory data
    # So first we load this data. Note that we only have info up to the Dec 2020. 
    CD = framework.calculate_days_of_weather(SM, SY, PD, "cold")  
    HD = framework.calculate_days_of_weather(SM, SY, PD, "hot")  
    RD = framework.calculate_days_of_weather(SM, SY, PD, "rain") 

    # Combining features into one input X
    X = [PT, AS, AT, OB, PD, SM, SY, SS, MW, EU, WU, HWB, IA, CD, HD, RD]
    
    # Predicting the target according to the problem, which is given in line 26 
    prediction = framework.predict(X, problem) 

    # Printing out the results for your information, you can get rid of this line
    print(f"Project's {problem} predicted to be ", ' '.join(str(round(x[0], 2)) for x in prediction))
   
   
if __name__=="__main__":
    main()