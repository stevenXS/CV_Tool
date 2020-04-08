### 自注：mmdetection中configs文件一般默认是使用imagenet预训练的backbone权重参数，但是使用coco预训练一般来说会使模型收敛更快，效果更好，是比赛提分的一个小trick！
###这里主要是修改coco预训练模型的类别不一致问题
# mmdet默认加载权重优先级别是resume_from(断点加载)，load_from，pretrained的顺序，所以需要从load_from加载预训练权重！

import torch

def main():
    #gen coco pretrained weight
    import torch
    #num_classes = 21
    num_classes = 16  # clw note: need modify
    model_coco = torch.load("/home/user/.cache/torch/checkpoints/cascade_rcnn_x101_64x4d_fpn_2x_20181218-5add321e.pth")
    #model_coco = torch.load("epoch_best.pth")  # clw test

    # weight
    model_coco["state_dict"]["bbox_head.0.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.0.fc_cls.weight"][
                                                            :num_classes, :]
    model_coco["state_dict"]["bbox_head.1.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.1.fc_cls.weight"][
                                                            :num_classes, :]
    model_coco["state_dict"]["bbox_head.2.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.2.fc_cls.weight"][
                                                            :num_classes, :]
    # bias
    model_coco["state_dict"]["bbox_head.0.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.0.fc_cls.bias"][
                                                          :num_classes]
    model_coco["state_dict"]["bbox_head.1.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.1.fc_cls.bias"][
                                                          :num_classes]
    model_coco["state_dict"]["bbox_head.2.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.2.fc_cls.bias"][
                                                          :num_classes]
    # save new model
    torch.save(model_coco, "cascade_rcnn_x101_64x4d_coco_pretrained_weights_classes_%d.pth" % num_classes)

if __name__ == "__main__":
    main()

