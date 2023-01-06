# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 07:26:49 2023

@author: valla
"""
import pandas as pd
def plot_success_rate(csv_file):
    
    data = pd.read_csv(csv_file)
    data
if __name__ == "__main__":
    plot_success_rate("..\experiment results\success_rate_OI")