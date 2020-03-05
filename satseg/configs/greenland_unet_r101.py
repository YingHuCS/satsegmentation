model = dict(
    name = 'UNet',
    backbone_name = 'resnet101', # resnet50 or resnet101 or resnext...
    pretrained = True,
    num_classes = 2)


data = dict(
    classes = ['background', 'greenland'],
    weights = [0.5, 1.75],
    train = dict(
        img_dir = '/raid/ying/datasets/greenland_seg/sliced_train_for_ad_images_new',
        mask_dir = '/raid/ying/datasets/greenland_seg/sliced_train_for_ad_masks_new',
        #followings are for data augmentation
        resize_scale = (0.8, 2.0),
        crop_size = 1024,
        flip_prob = 0.5,
        rotate_degree = (0, 10),
        rotation_prob = 0.5,
        rotation_degree = 90,
        img_mean = [0.485, 0.456, 0.406],
        img_std = [0.229, 0.224, 0.225]),
    val = dict(  
        img_dir = '/raid/ying/datasets/greenland_seg/sliced_train_for_ad_images_new',
        mask_dir = '/raid/ying/datasets/greenland_seg/sliced_train_for_ad_masks_new',
        img_mean = [0.485, 0.456, 0.406],
        img_std = [0.229, 0.224, 0.225]),
    test = dict(
        img_dir = '/home/ying/run/sliced_ours_4096',
        out_dir = '/home/ying/run/predictions_2017',
        img_mean = [0.485, 0.456, 0.406],
        img_std = [0.229, 0.224, 0.225]))


optimizer = dict(
    name = 'Adam',
    lr = 0.001,
    momentum = 0.9,
    weight_decay = 0.0001)


scheduler = dict(
    step_size = 80,
    gamma = 0.5)


loss = 'Mix' # Loss function name ('Lovasz', 'mIoU', 'CrossEntropy', 'Focal', 'Dice', 'Mix')
batch_size = 32
num_workers = 4
num_epochs = 300
use_gpu = True
checkpoint = '/raid/ying/checkpoints/satseg_greenland/checkpoint-00299-of-00300.pth'
#checkpoint = None
checkpoint_dir = '/raid/ying/checkpoints/satseg_greenland/train_for_ad_new'
resume = False

