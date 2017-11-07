"""
Author @Prakash

Create two classes
- Mopping (Contains all mopping videos) - 150 videos
- Others  (Contains 2 videos each from all other videos) - 200 videos

Generates a txt file with
vid_loc class_name

"""
import glob
import random
import os

files = glob.glob("../../UCF-101/*")
print("Total files:", len(files))
txt_loc = "txt_files/"

if not os.path.exists(txt_loc):
    os.makedirs(txt_loc)


obj_names = [i.rsplit("/")[-1] for i in files]
names = open(txt_loc+"obj.names", "w")

for i in obj_names:
    names.write(i+"\n")

train = open(txt_loc+ 'train.lst', 'w')
valid = open(txt_loc+ 'valid.lst', 'w')
test  = open(txt_loc+ 'test.lst', 'w')


for c in files:
    classs = c.rsplit("/")[-1]
    k = glob.glob(c+"/*.avi")
    random.shuffle(k)
    if classs in obj_names:
        #print("total files:", len(k))
        num1, num2 = int(0.6*len(k)), int(0.85*len(k))
        x_train = k[0:num1]
        x_valid = k[num1:num2]
        x_test = k[num2:]
        cat = classs
        for t in x_train:
            train.write(t+ " "+ cat+"\n")
        for v in x_valid:
            valid.write(v+ " "+ cat+"\n")
        for te in x_test:
            test.write(te+ " "+ cat+"\n")
