# written by clw

import xml.etree.ElementTree as ET
import os

xml_path = 'C:/Users/62349/Desktop/all/crop'
file_names = os.listdir(xml_path)

count_remove = 0
for file_name in file_names:
    if file_name.endswith('.png') or file_name.endswith('.jpg'):
        if not os.path.exists(os.path.join(xml_path, file_name.split('.')[0] + '.xml' )):
            os.remove(os.path.join(xml_path, file_name))
            count_remove += 1
            print('remove %d file: %s' % (count_remove, file_name) )

