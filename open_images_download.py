"""
required packages
pip install pandas urllib3 tqdm numpy awscli
"""
import platform
import os

import pandas as pd
from tqdm import tqdm

from multiprocessing.dummy import Pool as ThreadPool

# this is where you select where the data is downloaded from
data_type = "train"
# data_type = "validation"
#data_type = "test"


# modify these to point to the right locations
if platform.system() == "Windows":
    annotation_file_root = "D:/Projects/object_detection_data/"
    download_dir = "D:/Projects/object_detection_data/open_images/"
elif platform.system() == "Linux":
    home = os.path.expanduser('~')
    annotation_file_root = home + "/Projects/data/"
    download_dir = home + "/Projects/data/open_images/"
else:
    print("Quiting!")
    quit()

#class_names = ["/m/01940j", "backpack" ]     # backpack
class_names = ["/m/025dyy", "box"]     # box

# build the correct paths to save everything
#download_dir = "D:/Projects/object_detection_data/open_images/box"

# print out some of the important variables to double check things
print("Annotation Root: " + annotation_file_root)
print("Download Dir:    " + download_dir)
print("Data Type:       " + data_type)
print("Class Name:      " + class_names[1])


# read in the file
annotation_file = annotation_file_root + data_type + "-annotations-bbox.csv"
f = pd.read_csv(annotation_file)

# parse the file and then pull out the lines that match the class name
u = f.loc[f['LabelName'].isin([class_names[0]])]

u.to_csv(download_dir + data_type + "-" + class_names[1] + "-annotations-bbox.csv")

threads = 10
pool = ThreadPool(threads)

commands = []

# cycle through each entry and add to the commands list
for idx in u.index:
    image_filename = u['ImageID'][idx] + ".jpg"
    command = "aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/" + data_type + "/" + image_filename + " " + download_dir + class_names[1]
    print(command)
    commands.append(command)

# start downloading the images
list(tqdm(pool.imap(os.system, commands), total = len(commands) ))

print('Done!')
pool.close()
pool.join()

