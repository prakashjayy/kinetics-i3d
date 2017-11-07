""" Utils file
"""
import numpy as np
import cv2
import os
from tqdm import tqdm
import glob
import random

_LABELS = [i.split()[0] for i in open("txt_files/obj.names", "r")]



def gen_frame(nframes, frames=16, stride=8):
    """ Generate list of list acording to frames and stride
    """
    last_num = nframes - (nframes%stride)
    total_batches = int(last_num /stride)
    for i in range(total_batches-1):
        m = i * stride
        n = (frames) + (i * stride)
        x = list(range(m, n))
        yield x

def read_vid(vid_loc):
    cap = cv2.VideoCapture(vid_loc)
    vid = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        vid.append(cv2.resize(img, (455, 256)))
    vid = np.array(vid, dtype=np.float32)
    return vid

def create_npy_train_dataset(txt_loc, formats=".mp4", db_name="data/train/"):
    num_lines = sum(1 for line in open(txt_loc))
    for k, i in tqdm(enumerate(open(txt_loc, "r"))):
        classs = i.split()[1]
        loc = i.split()[0]
        vid = read_vid(loc)
        label = _LABELS.index(classs)
        for f, fra in enumerate(gen_frame(len(vid), 64, 32)):
            X = np.concatenate([vid[m][np.newaxis, :, :, :] for m in fra])
            X = X[np.newaxis, :, :, :, :]
            np.save(db_name+str(k)+"_"+str(f)+"_"+classs+".npy", X, allow_pickle=True)
        if k>100:
            break

def one_hot(batch_LABELS):
    nb_classes = len(_LABELS)
    targets = np.array([_LABELS.index(i) for i in batch_LABELS])
    one_hot_targets = np.eye(nb_classes)[targets]
    return one_hot_targets

def random_crop(array):
    num1 = np.random.randint(0, 256 - 224)
    num2 = np.random.randint(0, 455 - 224)
    crop = [ik[ :, :,  num1:num1+224, num2:num2+224, :] for ik in array]
    return  np.concatenate(crop)

def npy_reader(f, mean_file):
    labels = [i.rsplit("/")[-1].rsplit(".")[0].rsplit("_")[-1] for i in f]
    f = [np.load(i) - mean_file for i in f]
    f = random_crop(f)
    return (f, labels)

def npy_reader_valid(f, mean_file):
    labels = f.rsplit("/")[-1].rsplit(".")[0].rsplit("_")[-1]
    f = np.load(f) - mean_file
    f = f[ :, :,  16:240, 115:339, :]
    return (f, labels)

def test_vid(vid_loc, model_final, mean_file):
    vid = read_vid(vid_loc)
    vid = np.array(vid, dtype=np.float32)
    print(vid.shape)

    output = []
    for f in tqdm(gen_frame(vid.shape[0], 64, 32)):
        X = np.concatenate([vid[i][np.newaxis, :, :, :] for i in f])
        X = X[np.newaxis, :, :, :, :]
        X -= mean_file
        X = X[:, :, 16:240, 115:339, :]
        m = model_final.predict(X)
        output.append(m)

    output = np.concatenate(output)
    output1 = np.argmax(output.mean(axis=0))
    prob = np.max(output.mean(axis=0))
    return _LABELS[output1], prob

def mean_data(data_loc="data/train", sample=True, save_loc="data/mean_data__ucf.npy"):
    files = glob.glob(data_loc+"/*.npy")
    random.shuffle(files)
    files = files[0: int(0.1*len(files))]
    print("total_files: {}".format(len(files)))
    x = 0
    for k, i in enumerate(tqdm(files)):
        y = np.squeeze(np.load(i))
        y = np.float32(y)
        x = x+y

    final = x/k
    print(final)
    print(final.shape)

    np.save(save_loc, final, allow_pickle=True)



if __name__ == '__main__':
    if not os.path.exists("data/train/"):
        os.makedirs("data/train/")

    create_npy_train_dataset("txt_files/train.lst", ".mp4", db_name="data/train/")
    print(len(glob.glob("data/train/*.npy")))

    print("[Calculating Mean of the data]")
    mean_data()
