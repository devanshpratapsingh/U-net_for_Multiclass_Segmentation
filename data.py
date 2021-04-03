#importing required libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def process_data(data_path, file_path):
    df = pd.read_csv(file_path, sep=" ", header=None)
    names = df[0].values
    print(names)

    images = [os.path.join(data_path, f"images/{name}.jpg") for name in names]
    mask = [os.path.join(data_path, f"annotations/trimaps/{name}.png") for name in names]

    return images,mask

#to load and get the paths of the required datasets
def load_data(path):
    train_valid_path = os.path.join(path, "annotations/trainval.txt")
    test_path = os.path.join(path, "annotations/test.txt")

    train_x,train_y = process_data(path, train_valid_path) #x is for images, and y is for mask
    test_x,test_y = process_data(path, test_path)

    train_x, valid_x = train_test_split(train_x, test_size=0.2, random_state=42)
    train_y, valid_y = train_test_split(train_y, test_size=0.2, random_state=42)

    return (train_x,train_y), (valid_x, valid_y), (test_x, test_y)


if __name__ == "__main__":
    path = "E:/MIC_Projects/U-Net_Implementation/"
    (train_x,train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    print(f"Dataset: Train:{len(train_x)} - Valid:{len(valid_x)} - Test:{len(test_x)}")
