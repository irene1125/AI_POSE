#/DATA/train_val/train/camera : {Action_num}_{Actor_num}_{Camera_num}.json 
#                     /images : {Action_num}_{Actor_num}_{Camera_num}_{Frame_num}.jpg
#                     /labels : 3D_{Action_num}_{Actor_num}_{Frame_num}.json

# images 파일명에서 세번째 요소 제거한 list 생성 (중복허용- 중복처리해야됨)   srcImage = images의 action_num_ actor_num_ frame_num  (3번째 빠짐)
  # /DATA/train_val/train/labels/str1 파일 만들기
# /DATA/split/task04_train/labels/str1 : list 1개씩 꺼내 마지막 요소 없앤 것(join)/ i 앞에 3D_ 붙인것 ,
#                                  /DATA/train_val/train/labels/str1

import os
import splitfolders
import shutil
from distutils.dir_util import copy_tree # 파이썬 폴더 복사


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = os.path.join(ROOT_PROJECT_DIR, 'DATA')


TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'split/task04_train') # 비교할 대상
PROJECT_DATA_DIR = os.path.join(DATA_DIR, 'train_val') # split될 대상

if __name__ == '__main__':
    
    IM_DIR = os.path.join(PROJECT_DATA_DIR, 'val/images')
    yet_im_path = os.listdir(IM_DIR) #  /DATA/train_val/val/images 내부 디렉토리명

    LA_PROJ_DIR1 = os.path.join(PROJECT_DATA_DIR, 'val/labels')
    im_name = []
    num = 0
    before = ''
    image_list = []
    # images 파일명에서 세번째 요소 제거한 list 생성
    for i in yet_im_path :
        print(i)
        im_path = os.path.join(IM_DIR, i)
        #print(im_path)
        #print(os.listdir(im_path))
        im_name.extend(os.listdir(im_path))
        #print(im_name)
    print(im_name)
    print(len(im_name))
    
    for i in im_name : #image 이름들
        print(i)
        #s =i.split('_')
        #print(s)
        #ss = s[:2] + s[-1] #??? 세번째 원소 빼는 과정 
        ss = i[:9]+i[11:] #??? 세번째 원소 빼는 과정 
        ss = ss.rstrip('.jpg')
        #print(ss)
        if(before == ss):
            continue
        before = ss
        image_list.append(ss)
    #print(image_list)
    print(len(im_name))

    # 복붙 과정
    num = 0
    already  = []
    for i in image_list: # 세번쨰 요소 뺀 image list
        num += 1
        print(num)
        LA_PROJ_DIR2 = os.path.join(LA_PROJ_DIR1, i[:8])        
        #for a in already :
        if i[:8] not in already :
            os.makedirs(LA_PROJ_DIR2)
        already.append(i[:8])
        j = '3D_' + i + '.json'
        shutil.copy2(os.path.join(TRAIN_DATA_DIR,'labels',i[:8],j), LA_PROJ_DIR2)
        