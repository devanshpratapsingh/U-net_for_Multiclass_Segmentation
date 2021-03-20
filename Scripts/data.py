#importing required libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def process_data(data_path, file_path):
    df = pd.read_csv(file_path, sep=" ", header=None)
    print(df)

#to load and get the paths of the required datasets
def load_data(path):
    train_path = os.path.join(path, "annotations/trainval.txt")
    test_path = os.path.join(path, "annotations/test.txt")

    train_x,train_y = process_data(path, train_path) #x is for images, and y is for mask
    test_x,test_y = process_data(path, test_path)

if __name__ == "__main__":
    path = "E:/MIC_Projects/U-Net_Implementation/"
    load_data(path)
