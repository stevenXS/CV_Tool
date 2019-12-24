import os
from PIL import Image

img_path = 'E:/textile/raw/20191017(deprecated)/image/'

from_format = '.png'
to_format = '.jpg'

files = os.listdir(img_path)
for idx, f in enumerate(files):
    x = os.path.splitext(f)
    # 判断jpg文件
    if x[1] == from_format:
        # 重命名
        im = Image.open(os.path.join(img_path,  x[0] + from_format))
        im.save(os.path.join(img_path, x[0] + to_format))
        os.remove(os.path.join(img_path,  x[0] + from_format))
        print('clw: idx=%d, 成功将文件%s转为%s格式' % (idx+1, x[0] + from_format, to_format))