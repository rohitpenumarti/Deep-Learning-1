import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def read_and_clean_data_file(file):
    with open(file) as f:
        lines = f.readlines()

    return lines

def main():
    lines = read_and_clean_data_file('cluster.txt')
    print(lines)

if __name__ == "__main__":
    main()