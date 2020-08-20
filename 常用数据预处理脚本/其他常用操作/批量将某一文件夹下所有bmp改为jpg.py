import os
import cv2

img_path = 'D:/pictures_forth_4tiaoguang_20200819'

for root, dirs, files in os.walk(img_path):
    for file_name in files:
        print('root: %s, file_name: %s' % (root, file_name))
        if file_name.endswith('.bmp'):
            img = cv2.imread(os.path.join(root, file_name))
            cv2.imwrite(os.path.join(root, file_name[:-4] + '.jpg'), img, [int( cv2.IMWRITE_JPEG_QUALITY), 100])
            os.remove(os.path.join(root, file_name))