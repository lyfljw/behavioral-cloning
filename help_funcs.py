import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

# @open csv file
def open_file(csv_name):
    """
    open the csv_file and return the list of each line in the file, and the training set size.
    This function helps to avoid rereading csv file when generate the batches.

    :param
        csv_name: the csv file path and name
    :return:
        lines: each line in the csv file,
                lines[0]: center img path;
                lines[1]:left image path;
                lines[2]: right image path;
                lines[3]: steering info
    """
    print("reading and pre-processing images...")
    lines = []
    with open("./" + csv_name) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines[1:-1]   # first line is description

def read_rgb(file_name):
    """
    read an image in RBG channel
    :param
        file_name:  a string of the path of the image file
    :return:
        a numpy array of the image in RGB mode.
    """
    image = cv2.imread(file_name)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# @random_trans
def random_trans(image, steer):
    """
    randomly translate an image in a few ways:
        -1: a 50% chance to flip the image

        0 do nothing
        1 adding gamma
        2 add bright
        3 shift horizontally and/or vertically
        4 add dark mask
    :param
        image: the processed-image after pre_process method (66,200,3)
        steer: the original steer of the image

    :return:
        image: the transformed image
        steer: the new steer
    """
    trans = np.random.randint(5)
    if np.random.randint(1)==0:
        image = cv2.flip(image, 1)
        steer = -steer

    if trans == 0:
        return image, steer
    elif trans == 1:
        gamma = np.random.uniform(0.4, 1.5)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table), steer

    elif trans == 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image = np.array(image, dtype=np.float64)
        random_bright = 0.5+np.random.uniform()
        image[:,:,2] = image[:,:,2]*random_bright
        image[:, :, 2][image[:, :, 2] > 255] = 255
        image = np.array(image, dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image, steer

    elif trans == 3:
        trans_x = 100 * (np.random.rand() - 0.5)
        trans_y = 10 * (np.random.rand() - 0.5)
        steer += trans_x * 0.002
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = image.shape[:2]
        image = cv2.warpAffine(image, trans_m, (width, height))
        return image, steer

    elif trans ==4:
        x1, y1 = 200 * np.random.rand(), 0
        x2, y2 = 200 * np.random.rand(), 66
        xm, ym = np.mgrid[0:66, 0:200]
        mask = np.zeros_like(image[:, :, 1])
        mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1
        cond = mask == np.random.randint(2)
        s_ratio = np.random.uniform(low=0.2, high=0.5)
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB), steer

# @generate the data batch
def get_training_data(lines):
    """
    Parameters:
        lines: each line in the csv file, from the open_file method
    return:
        X_train: numpy array images
        y_train: controller data
    """
    examples = len(lines)
    images = []
    steers = []
    for line in lines:
        file_name = line[0].lstrip()
        steer = float(line[3])
        image = read_rgb(file_name)
        image = pre_process(image)
        images.append(image)
        steers.append(steer)
        # random trans, add 2 more images
        image, steer = random_trans(image, steer)
        images.append(image)
        steers.append(steer)
        image, steer = random_trans(image, steer)
        images.append(image)
        steers.append(steer)
        ## left images
        file_name = line[1].lstrip()
        steer = float(line[3])+0.2
        image = read_rgb(file_name)
        image = pre_process(image)
        images.append(image)
        steers.append(steer)
        # random trans, add 2 more images
        image, steer = random_trans(image, steer)
        images.append(image)
        steers.append(steer)
        image, steer = random_trans(image, steer)
        images.append(image)
        steers.append(steer)

        ## right images
        file_name = line[2].lstrip()
        steer = float(line[3]) - 0.2
        image = read_rgb(file_name)
        image = pre_process(image)
        images.append(image)
        steers.append(steer)
        # random trans, add 2 more images
        image, steer = random_trans(image, steer)
        images.append(image)
        steers.append(steer)
        image, steer = random_trans(image, steer)
        images.append(image)
        steers.append(steer)
    return shuffle(np.array(images), np.array(steers))


def pre_process(image):
    """
    pre-process an image by cropping, resize

    :param image: original image with shape of (160 320 3)
    :return: an (66,200,3) image
    """
    image = image[60:-25,:,:]
    image = cv2.resize(image, (200, 66), cv2.INTER_AREA)
    return image

def read_valid(csv_file):
    """
    read a csv file generate by the simulator
    :param
        csv_file: a string of the path of the file name
    :return:
        images and steering angles
    """
    lines = open_file(csv_name=csv_file)
    examples = len(lines)
    X = np.empty([examples*3,66,200,3])
    y = np.empty(examples*3)
    for i in range(examples):
        X[i] = pre_process(cv2.imread(lines[i][0]))
        X[i+examples] = pre_process(cv2.imread(lines[i][1]))
        X[i + 2*examples] = pre_process(cv2.imread(lines[i][2]))
        steer = float(lines[i][3])
        y[i] = steer
        y[i+examples] = steer+0.2
        y[i+2*examples] = steer-0.2

    return shuffle(X, y)

