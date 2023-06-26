import os
import shutil

source_dir = '../../dataset/aug_data/val/'
image_dest_dir = '../../dataset/images/val/'
label_dest_dir = '../../dataset/labels/val/'

for filename in os.listdir(source_dir):
    if filename.endswith('.jpg'):  
        shutil.move(os.path.join(source_dir, filename), os.path.join(image_dest_dir, filename))
    elif filename.endswith('.txt'):
        shutil.move(os.path.join(source_dir, filename), os.path.join(label_dest_dir, filename))