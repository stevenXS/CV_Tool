# 功能： 1、只提取所有正样本
#        2、.bmp转成.jpg
#        3、resize到608x608，
#        保存到一个新的文件夹
#        如果resize，就暂不拷贝xml，因为xml里面保存的是实际尺寸而不是txt的比例尺寸，因此坐标还需要修正；
#        xml文件主要就起到一个可视化的作用，训练时并不会用到；

import shutil
import os
import cv2
from tqdm import tqdm

data_path = 'D:\dataset\HRSC2016_dataset\HRSC2016\Train\AllImages'
save_path = 'D:\dataset\HRSC2016_dataset\HRSC2016\Train\hrsc2016'

# data_path = 'D:\dataset\HRSC2016_dataset\HRSC2016\Test\AllImages'
# save_path = 'D:\dataset\HRSC2016_dataset\HRSC2016\Test\hrsc2016'

if not os.path.exists(save_path):
    os.makedirs(save_path)

txt_file_list = [str for str in os.listdir(data_path) if str.endswith('.txt')]
for txt_file_name in tqdm(txt_file_list):
    with open(os.path.join(data_path, txt_file_name), 'r') as f:
        lines = f.readlines()
        if len(lines) == 0:
            continue
        shutil.copy(os.path.join(data_path, txt_file_name),  os.path.join(save_path, txt_file_name))
        ### clw note: resize后，之前的xml里面的box就不对了，所以暂时先不考虑复制xml
        #shutil.copy(os.path.join(data_path, txt_file_name[:-4]+'.xml'),  os.path.join(save_path, txt_file_name[:-4]+'.xml'))
        img_file_name = txt_file_name[:-4] + '.bmp'
        img = cv2.imread(os.path.join(data_path, img_file_name))
        img = cv2.resize(img, (608, 608), interpolation = cv2.INTER_AREA)
        cv2.imwrite(os.path.join(save_path, img_file_name[:-4] + '.jpg'), img)
