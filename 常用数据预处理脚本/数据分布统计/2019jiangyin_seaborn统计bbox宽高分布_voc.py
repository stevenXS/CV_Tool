import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['figure.figsize'] = (10.0, 10.0)


xmls_path = '/media/clwclw/data/2019jiangyin/train_classes/joint2/'
images_file = '/media/clwclw/data/2019jiangyin/train_classes/joint2/'

import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

### seaborn绘制bbox框高尺寸图
xml_names = os.listdir(xmls_path)

categorys_w = []
categorys_h = []
xml_names = [xml_name for xml_name in xml_names if xml_name.endswith('.xml')]
for xml_name in xml_names:
    print('xml_name:', xml_name)
    tree = ET.parse(os.path.join(xmls_path, xml_name))
    root = tree.getroot()
    for anno_id, obj in enumerate(root.iter('object')):
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        w = xmax - xmin
        h = ymax - ymin
        categorys_w.append(w)
        categorys_h.append(h)

data_list = list(zip(categorys_w, categorys_h))
data = np.array(data_list)

### 或者随机生成一些数据
# np.random.seed(sum(map(ord, "distributions")))
# mean, cov = [0, 1], [(1, .5), (.5, 1)]
# data = np.random.multivariate_normal(mean, cov, 200)




df = pd.DataFrame(data, columns=["x", "y"])

sns.set(color_codes=True)
g = sns.jointplot(x="x", y="y", data=df, kind="kde", color="m")
g.plot_joint(plt.scatter, c="m", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)  #画背景网格线
g.set_axis_labels("$X$", "$Y$")
plt.show()


