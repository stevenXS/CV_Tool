针对labelme的json数据格式、yolov3的txt数据格式､voc的xml数据格式，编写json2txt.py、json2xml.py、txt2xml.py,支持互转：
txt2json.py, 支持yolov3的txt数据格式转为labelme的json数据格式；
xml2json.py, voc的xml数据格式转为labelme的json数据格式；
xml_size_fmt.py，支持将各种不同大小的标签数据转换为指定的大小(如1000x1000)的标签数据，并根据标签的最大尺寸及最小尺寸再调整缩放比例；

