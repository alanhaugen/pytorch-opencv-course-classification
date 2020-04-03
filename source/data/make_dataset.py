#!/usr/bin/python

import os
os.system('kaggle competitions download -c pytorch-opencv-course-classification')
os.system('mkdir data')
os.system('unzip pytorch-opencv-course-classification -d data')
os.system('rm pytorch-opencv-course-classification')

