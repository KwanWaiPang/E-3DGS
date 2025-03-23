import os
import glob
import cv2
import sys

dir_name = sys.argv[1]

file_names = glob.glob(os.path.join(dir_name, "*.png"))

for file_name in file_names:
    img = cv2.imread(file_name)
    cv2.imwrite(file_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))