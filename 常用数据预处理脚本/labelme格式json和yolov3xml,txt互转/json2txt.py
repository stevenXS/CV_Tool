#coding:utf-8

## reference
## https://blog.csdn.net/x779250919/article/details/103927525

import os
import sys
import json
import numpy as np
import shutil


class2indice = {"car":0, "suv":0, "van":0, "bus":0, "truck":0, "inner":1, 'outter': 2}

def json2txt(path_json, path_txt):
    print("process", path_json)
    with open(path_json,'r') as path_json:
        jsonx=json.load(path_json)
        img_w = jsonx["imageWidth"]
        img_h = jsonx["imageHeight"]
        with open(path_txt,'w+') as ftxt:
            for shape in jsonx['shapes']:
                points = np.array(shape['points'])
                x1, y1=points[0]
                x2, y2=points[1]
                label_w = abs(x2 - x1)
                label_h = abs(y2 - y1)
                label=str(shape['label'])
                ci = class2indice[label]
                strxy = str(ci) + " " + str((x1+label_w/2)/ img_w) + ' ' + str((y1+label_h/2)/ img_h) + " " + str(label_w/ img_w) + ' ' + str(label_h/ img_h)
                ftxt.writelines(strxy+"\n")  
    pass

def process(image_json_path, output_imgf_path):
    lines = []
    abs_image_path = os.path.abspath( image_json_path )
    for pos,_,fs in os.walk( abs_image_path ):
        for f in fs:
            fmt = f[-4:].lower()
            if fmt == '.jpg' or fmt == '.png':
                imgf_path = os.path.join(pos, f)
                json_file_path = imgf_path[:-4] + ".json"

                if os.path.exists(json_file_path):
                    txt_file_path = imgf_path[:-4] + ".txt"
                    json2txt(json_file_path, txt_file_path)
                    lines.append( imgf_path )
                else:
                    print("miss", json_file_path)
    open(output_imgf_path, 'w').write( '\n'.join(lines) )
    pass


def main(train_path = 'train/', valid_path = 'val/'):
    if train_path is not None and os.path.exists(train_path):
        process(train_path, './train.txt')
    else:
        print("not exists '%s'" %(train_path))

    if valid_path is not None and os.path.exists(valid_path):
        process(valid_path, './valid.txt')
    else:
        print("not exists '%s'" %(valid_path))
    pass


if __name__ == "__main__":
    train_path = "train/"
    val_path = "val/"

    if len(sys.argv) > 1:
        train_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        val_path = sys.argv[2]

    main(train_path, val_path)
    pass
