# 根据某个文件夹下的所有文件，在其他文件夹找到同名的文件
# 比如src下的图片都是绘制了检测框的，在dst下把所有原图找回来


import os
import argparse
import shutil

JPG_EXT = '.jpg'

def get_src_file(file_path):
    src_files = []
    for pos, _, files in os.walk(file_path):
        for file in files:
            if file.endswith(JPG_EXT):
                src_files.append(file[:-4])
    return set(src_files)

def cp_dst_file(src_files, dst_path, save_path):
    find_files = []
    for pos, _, files in os.walk(dst_path):
        print('clw: find at path: ', pos)
        for file in files:
            if file[:-4] in src_files:
                find_files.append(file)
                shutil.copy(os.path.join(pos, file), save_path)
    if len(src_files) != len(find_files):
        print('clw: {} files not find !!!'.format(len(src_files) - len(find_files)) )
    else:
        print('clw: find {} files all.'.format(len(src_files)))

def copy_file(src_path, dst_path, save_path):
    if not os.path.isdir(src_path):
        return
    if not os.path.isdir(dst_path):
        return
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('start to get_src_file: ')
    src_files = get_src_file(src_path)
    print('start to cp_dst_file: ')
    cp_dst_file(src_files, dst_path, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src_path', type=str, help='src path')
    parser.add_argument('dst_path', type=str, help='dst path')
    parser.add_argument('save_path', type=str, help='save path')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    copy_file(opt.src_path, opt.dst_path, opt.save_path)