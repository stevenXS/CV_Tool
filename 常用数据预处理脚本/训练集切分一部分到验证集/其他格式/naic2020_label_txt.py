'''
输入：
images -> *.png
label.txt

输出：
train.txt
val.txt
query.txt
gallery.txt

其中，label.txt格式如下
00040591.png:15178
00066284.png:15178
00025569.png:15178
00024054.png:15178
00028221.png:4664
00013144.png:4664
00048709.png:4664
00011252.png:4664
00021904.png:5044
00045102.png:5044
00064956.png:5044
00053668.png:5044
00001270.png:12102
00057448.png:12102
00065274.png:12102
00011815.png:12102
......
正样本: 00000000.png~00072823.png,72824个（包含4151张只有自己的，无法配对，这种也作为负样本）
pid: 0~19657
负样本：00072824~00093601，20778个
验证集的query不要用只有一张图的这种；
训练集也不要用只有一张图的这种；

验证集划分：
考虑只有自己一张图的都放到验证集的gallery，也就是  20778 + 4151 = 24929 张，作为验证集的负样本集合，放到gallery；
剩下的随机往验证集的gallery和query扔，但是相同pid的，确保query至少有一张
由于测试集gallery和query比例14:1，那么验证集最好也是这个比例；
相同pid有4张的这种情况最多，有10837组，
如果把2张扔到gallery，2张扔到query，那么只需要 959 组，就可以达到接近14:1的比例；

总结：
train: 72824 - 4151 - 959*2 = 66755
val = query: 959*2 = 1918
gallery: 20778+4151=24929

'''

import os

# 1、读取txt，用dict记录 imname 及 对应的pid
imname_pid_dict = {}
pid_nums_count_dict = {}
count_pid_dict = {}
root = 'D:/dataset/2020NAIC-REID/train'
label_path = os.path.join(root, 'label.txt')
pos_count = 0

same_count = 0
image_names= os.listdir(os.path.join(root, 'images'))

val = ""
train = ""
query = ""
gallery = ""

with open(label_path) as f:
    lines = f.readlines()
    pre_pid = lines[0].strip().split(':')[1]
    for i, line in enumerate(lines):
        pos_count += 1
        l, r = line.strip().split(':')
        print(i, l)

        if l not in imname_pid_dict:
            imname_pid_dict[l] = r
        else:
            raise Exception('Error: already have key %s' % l)


        if r!= pre_pid:  # 名字不同，记录总共有几张这个人的图片； 必须是顺序
            if same_count not in pid_nums_count_dict :
                pid_nums_count_dict[same_count] = 1       # 只有自己本身也就是1张图(key)的，有4151张(value)，其实也应该作为负样本；
            else:
                pid_nums_count_dict[same_count] += 1

            if same_count not in count_pid_dict:
                count_pid_dict[same_count] = [ pre_pid ]  # 最多的有749张图(key)，对应的pid是1107(value)
            else:
                count_pid_dict[same_count].append(pre_pid)


            pre_pid = r
            same_count = 1
        else:
            same_count += 1

    for i in range(72824, 93601+1):  # 负样本 00072824~00093601
        gallery += ('000' + str(i) + '.png' +'\n')

    ### 第2次遍历，把验证集选出来
    val_count = 0
    same_pid_count = 0
    b_needVal = True
    for i, line in enumerate(lines):
        l, r = line.strip().split(':')
        print(i, l)
        if r in count_pid_dict[4]:  # 相同pid有4张的这种情况最多，有10837组， 如果把2张扔到gallery，2张扔到query，那么只需要 959 组，就可以达到接近14:1的比例；
            if b_needVal:
                if same_pid_count >= 4:  # 把之前pid的先恢复了
                    same_pid_count = 0
                if same_pid_count >= 2:  # 2,3 加入验证集
                    val += line
                    query += l+'\n'
                    val_count += 1
                    same_pid_count += 1
                else:  # 0,1
                    train += line
                    same_pid_count += 1

                if val_count >= 959 * 2:
                    b_needVal = False
            else: # 验证集已经加满，剩下也全部加入训练集
                train += line
        elif r in count_pid_dict[1]:    # 只有自己一张图的，也都放到验证集的gallery
            gallery += l+'\n'
        else: # 全部加入训练集
            train += line

    # 训练集70906 + 验证集1918 = 72824
    with open(os.path.join(root, 'train.txt'), 'w') as f:
        f.write(train)
    with open(os.path.join(root, 'val.txt'), 'w') as f:
        f.write(val)
    with open(os.path.join(root, 'query.txt'), 'w') as f:
        f.write(query)
    with open(os.path.join(root, 'gallery.txt'), 'w') as f:
        f.write(gallery)

    print('end !!')


neg_count = len(image_names) - pos_count


# 2、切分验证集；比例最好和线上一致；线上是 14：1
#    这里把 20778张负样本扔到验证集的gallery，然后




print('end!')


