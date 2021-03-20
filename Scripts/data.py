#importing required libraries
import os
import numpy as np
import pandas as py
from sklearn.model_selection import train_test_split


#to load and get the paths of the required datasets
def load_data(path):
  train_path = os.path.join(path, "annotations/trainval.txt")
  test_path = os.path.join(path, "annotations/test.txt")

  train_x,train_y = process_data(path, train_path) #x is for images, and y is for mask
  test_x,test_y = process_data(path, test_path)