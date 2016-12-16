# This script is for combining all the test data

# import modules
import numpy as np
import pandas as pd
from os import listdir

def testcombine():

    # load test files and combine
    mypath = "../Data"
    test_names = [f for f in listdir(mypath) if "test" in f]
    test_files = [pd.read_csv("../Data/%s"  % file, header = None) for file in test_names]
    test = pd.concat(test_files, axis = 0, ignore_index = True)

    # export to csv
    test.to_csv("../Data/blogData_test.csv",index=False,header=False)

if __name__ == "__main__":
    testcombine()
    print("generate test file")
