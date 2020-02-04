import os
import argparse
import xml.etree.ElementTree as ET
import shutil

def get_xml_files(xml_root_path):
    xml_files = []
    if not os.path.isdir(xml_root_path):
        return xml_files
    for root, _, files in os.walk(xml_root_path):
        for file in files:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))
    return xml_files

def copy_file(xml_file, dst):
    if not os.path.exists(dst):
       os.makedirs(dst)
    img_file = xml_file[:-4] + '.jpg'
    if os.path.isfile(xml_file) and os.path.isfile(img_file):
        shutil.copy(xml_file, dst)
        shutil.copy(img_file, dst)

def classify_picture(xml_root_path):
    if not os.path.isdir(xml_root_path):
        print("invalid xml root path")
        return
    output_path = "./classify/"
    if not os.path.exists(output_path):
       os.makedirs(output_path)

    xml_files = get_xml_files(xml_root_path)
    for xml_file in xml_files:
        copyed = {}
        tree = ET.parse(xml_file)
        root = tree.getroot()
        objs = root.findall('object')
        for obj in objs:
            name = obj.find('name').text
            if name not in copyed:
                copyed[name] = 1
                dst = output_path + name
                copy_file(xml_file, dst)


if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('xml_root_path', type=str, help='root path contains xml label')
    #opt = parser.parse_args()
    #print(opt, end='\n\n')

    #classify_picture(opt.xml_root_path)
    classify_picture('/media/clwclw/data/2019xxxx/train/')
