model = dict(
    name = 'UNet',
    backbone_name = 'resnet101', # resnet50 or resnet101 or resnext...
    pretrained = True,
    num_classes = 2)


data = dict(
    classes = ['background', 'road'],
    weights = [0.1, 1.75],
    train = dict(
        img_dir = '/raid/ying/datasets/road_seg/trainval/images',
        #img_dir = '/home/ying/temp/images',
        mask_dir = '/raid/ying/datasets/road_seg/trainval/masks',
        #mask_dir = '/home/ying/temp/masks',
        #followings are for data augmentation
        resize_scale = (1.0, 2.0),
        crop_size = 1024,
        flip_prob = 0.5,
        rotate_degree = (0, 360),
        rotation_prob = 0.5,
        rotation_degree = 90,
        img_mean = [0.485, 0.456, 0.406],
        img_std = [0.229, 0.224, 0.225]),
    val = dict(  
        img_dir = '/raid/ying/datasets/road_seg/validation/images',
        #img_dir = '/home/ying/temp/images',
        mask_dir = '/raid/ying/datasets/road_seg/validation/masks',
        #mask_dir = '/home/ying/temp/masks',
        img_mean = [0.485, 0.456, 0.406],
        img_std = [0.229, 0.224, 0.225]),
    test = dict(
        img_dir = '/home/ying/run/sliced_ours_multiscales/',
        out_dir = '/home/ying/run/predictions',
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
num_epochs = 500
use_gpu = True
checkpoint = '/satsegmentation/trained_checkpoints/Mix_Adam_32_001_330.pth'
#checkpoint = None
checkpoint_dir = ''
resume = False

