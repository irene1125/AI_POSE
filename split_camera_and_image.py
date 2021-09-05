import os
import splitfolders
from distutils.dir_util import copy_tree # 파이썬 폴더 복사

# CONFIG
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = os.path.join(ROOT_PROJECT_DIR, 'DATA') 
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'split/task04_train') # split할 data
PROJECT_DATA_DIR = os.path.join(DATA_DIR, 'train_val') # split 들어가는 곳
print(DATA_DIR) # 확인용
print(TRAIN_DATA_DIR) # 확인용
TRAIN_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/train_config.yml')


if __name__ == '__main__':
    
    splitfolders.ratio(TRAIN_DATA_DIR, output=PROJECT_DATA_DIR, seed=77, ratio=(0.8, 0.2))

    CAM_DIR = os.path.join(PROJECT_DATA_DIR, 'train/camera')
    IM_DIR = os.path.join(TRAIN_DATA_DIR, 'images')
    filename = os.listdir(CAM_DIR)
    im_name = os.listdir(IM_DIR)
    
    VAL_CAM_DIR = os.path.join(PROJECT_DATA_DIR, 'val/camera')
    val_filename = os.listdir(VAL_CAM_DIR)
    
   
    filenames = [f.rstrip('.json') for f in filename if f.endswith('.json')]
    val_filenames = [f.rstrip('.json') for f in val_filename if f.endswith('.json')]
    j = 0
    for i in filenames:
        if i in im_name:
            j +=1
            print(j)
            des = os.path.join(PROJECT_DATA_DIR,'train/images')
            os.makedirs(os.path.join(des,i))
            copy_tree(os.path.join(IM_DIR,i), os.path.join(des,i))
    print("train done")
    j = 0        
    for i in val_filenames:
        if i in im_name:
            j +=1
            print(j)
            des = os.path.join(PROJECT_DATA_DIR,'val/images')
            os.makedirs(os.path.join(des,i))
            copy_tree(os.path.join(IM_DIR,i), os.path.join(des,i))
    print("val done")
            
            
            