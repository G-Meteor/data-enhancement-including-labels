​
一、文档：
GitHub: https://github.com/albumentations-team/albumentations

官方文档：Albumentations Documentation

二、Installation
pip install -U albumentations

三、Keypoints augmentation Example:
1. Import the required libraries
import albumentations as A

import cv2

2. Define an augmentation pipeline.
transform = A.Compose([

    A.RandomCrop(width=330, height=330),

    A.RandomBrightnessContrast(p=0.2),

],keypoint_params=A.KeypointParams(format='xy',label_fields=['class_labels'], remove_invisible =False))

#对图片的操作放在第一个参数的列表中，可支持多种操作，p参数表示进行该操作的概率。

#A.KeypointParams()中，format参数定义ground_truth的坐标点格式，还有“yx”、“xysa”等；label_fields定义ground_truth中标签的名称,此参数是一个列表，可以设置多个标签；remove_invisible参数用来设置是否移除数据增强后超出图片范围的点，默认为True，若为True则不显示不可见的点，若为False则显示不可见的点

3. Read images, keypoints and class_labels
image = cv2.imread("/path/image.jpg")

#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

keypoints = [

    (264, 203),

    (86, 88),

    (254, 160),

    (193, 103),

    (65, 341),

]

#关键点信息以坐标的形式放入一个列表中，格式与format参数设置的格式同步

class_labels = [

    'left_elbow',

    'right_elbow',

    'left_wrist',

    'right_wrist',

    'right_hip',

]

4. Pass an image and keypoints to the augmentation pipeline and receive augmented images and points.
#对数据进行变换

transformed = transform(image=image, keypoints=keypoints, class_labels=class_labels)  

#读取变换后的数据

transformed_image = transformed['image']

transformed_keypoints = transformed['keypoints']

transformed_class_labels = transformed['class_labels']

四、List of augmentations
1、Spatial-level transforms (空间变换)
（1）、仿射变换

Affine(scale=None, translate_percent=None, translate_px=None, rotate=None, shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, always_apply=False, p=0.5) -support keypoints
#仿射变换

该变换包括Translation（移动）、Rotation（旋转）、Scaling（缩放）、Shear（剪切为梯形），其中：

scale参数，表示缩放的比例，支持数字、元组、字典；若为元组或字典，则从多个数字中等概率的取一个值对数据进行对应处理

translate_percent参数，按比例进行移动（x、y轴上都进行移动）

translate_px参数，表示移动像素点的距离（x、y轴上都进行移动），仅支持整数类型

rotate参数，表示旋转度数（不是弧度），范围为[-360, 360]，以图像中心为旋转点

shear参数，表示以度为单位进行剪切，预期值为[-360, 360]，合理值范围为[-45, 45]，如果是一个数字，则只进行x轴上的剪切；如果是元组(a,b)，则分别在x轴与y轴上进行剪切

cval参数，填充新创建的像素时使用的常量值

fit_output参数，是否使变换后的整个输出图像始终包含在图像平面中，默认为False：接受图像平面之外的图像，若为True的话：则只包含在图像平面中

p参数，表示应用变换的概率，默认值：0.5。

Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, mask_pad_val=0, fit_output=False, interpolation=1, always_apply=False, p=0.5) (support keypoints)
#随机四点透视变换

其中：

scale参数，表示正态分布的标准差，用于对子图像的角与完整图像的角的随机距离进行采样。默认值：(0.05, 0.1)，也可自己设置范围[float,float]

keep_size参数，是否在应用透视变换后将图像调整回其原始大小。默认值：真，如果设置为 False，生成的图像可能最终具有不同的形状。

fit_output参数，如果为 True，将调整图像平面大小和位置以在透视变换后仍捕获整个图像，默认值：假。

p参数，应用变换的概率。默认值：0.5。

PiecewiseAffine(scale=(0.03, 0.05), nb_rows=4, nb_cols=4, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode='constant', absolute_scale=False, always_apply=False, keypoints_threshold=0.01, p=0.5) (support keypoints)
#局部区域进行仿射变换

ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4, value=None, mask_value=None, shift_limit_x=None, shift_limit_y=None, always_apply=False, p=0.5) (support keypoints)
#随机应用仿射变换：平移、缩放和旋转

其中：

shift_limit 参数，平移操作的长宽限制，默认为 (-0.0625, 0.0625)

scale_limit 参数，缩放操作的范围限制，默认为 (-0.1, 0.1)

rotate_limit 参数，旋转操作的范围限制，默认为(-45,45)

p 参数，应用此变换的概率，默认为0.5

ElasticTransform #图像弹性变形
（2）、裁剪

Crop(x_min=0, y_min=0, x_max=1024, y_max=1024, always_apply=False, p=1.0) (support keypoints)
#裁剪自定义区域

x_min参数：左上角x坐标

y_min参数：左上角y坐标

x_max参数：右下角x坐标

y_max参数：右下角y坐标

CenterCrop(height, width, always_apply=False, p=1.0) (support keypoints)
#裁剪中心区域

height参数：裁剪的高度

width参数：裁剪的宽度

p参数：执行此操作的概率，默认为1

CropAndPad(px=None, percent=None, pad_mode=0, pad_cval=0, pad_cval_mask=0, keep_size=True, sample_independently=True, interpolation=1, always_apply=False, p=1.0) (support keypoints)
#裁剪/填充

px参数，要裁剪/填充的像素数，负数为裁剪，正数为填充；

percent参数，要裁剪/填充的比例，同样负数为裁剪，正数为填充，范围为[-1,1]；

keep_size 参数，表示是否保持输入图像的大小，默认为True。

CropNonEmptyMaskIfExists (height, width, ignore_values=None, ignore_channels=None, always_apply=False, p=1.0) (support keypoints)
#如果Mask非空，则使用掩码裁剪区域，否则进行随机裁剪

height参数：高的大小（以像素为单位）

width参数：宽的大小（以像素为单位）

RandomCrop (height, width, always_apply=False, p=1.0) (support keypoints)
#根据设置的长宽随机剪裁

RandomCropNearBBox(max_part_shift=(0.3,0.3), cropping_box_key='cropping_bbox', always_apply=False, p=1.0) (support keypoints)
#随机剪裁矩形框

RandomResizedCrop(height, width, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, always_apply=False, p=1.0) (support keypoints)
#随机裁剪并调整大小为height、width

RandomSizedCrop(min_max_height, height, width, w2h_ratio=1.0, interpolation=1, always_apply=False, p=1.0)(support keypoints)
#随机剪裁并缩放到定义大小

min_max_height参数：表示裁剪高的大小限制，格式为[int，int]

height参数：剪裁并调整大小后的高

width参数：参数：剪裁并调整大小后的宽

w2h_ratio参数：剪裁比例，默认为1.0

p参数：表示应用此变换的概率，默认为1.0

（3）、翻转

Flip.apply (self, img, d=0, **params) (support keypoints)
#翻转，包括水平翻转、垂直翻转、水平垂直同时进行翻转

d参数：0表示进行垂直翻转，1表示进行水平翻转，-1表示同时进行水平与垂直翻转

p参数：表示应用此变换的概率，默认为0.5

HorizontalFlip (support keypoints)
#水平翻转

p参数：表示应用此变换的概率，默认为0.5

VerticalFlip (support keypoints)
#垂直翻转

p参数：表示应用此变换的概率，默认为0.5

（4）、旋转

RandomRotate90 (support keypoints)
#随机旋转90度0次或多次

p参数：表示应用此变换的概率，默认为0.5

Rotate (limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5)(support keypoints)
#从定义的范围中随机旋转一个角度

limit参数：表示旋转的范围限制，范围为（-limit,limit），默认为（-90,90）

p参数：表示应用此变换的概率，默认为0.5

SafeRotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5)(support keypoints)
#从定义的范围中随机旋转一个角度

（5）、缩放

LongestMaxSize(max_size=1024, interpolation=1, always_apply=False, p=1) (support keypoints)
#重新缩放图像，使最大边等于 max_size，保持初始图像的纵横比。

RandomScale(scale_limit=0.1, interpolation=1, always_apply=False, p=0.5) (support keypoints)
#随机缩放

scale_limit参数：表示缩放范围，为（1-scale_limit,1+scale_limit）,默认为（0.9,1.1）

p参数：表示应用此变换的概率，默认为0.5

Resize (height, width, interpolation=1, always_apply=False, p=1)(support keypoints)
#调整为给定的大小

SmallestMaxSize(max_size=1024, interpolation=1, always_apply=False, p=1)(support keypoints)
#缩放图像，将最小边等于max_size

（6）、填充

PadIfNeeded(min_height=1024, min_width=1024, pad_height_divisor=None, pad_width_divisor=None, position=<PositionType.CENTER: 'center'>, border_mode=4, value=None, mask_value=None, always_apply=False, p=1.0) (support keypoints)
#如果图片大小小于需要的大小，可以对图片进行填充

（7）、其他

Lambda(image=None, mask=None, keypoint=None, bbox=None, name=None, always_apply=False, p=1.0) (support keypoints)
#可自定义的变换函数

image参数：图像变换函数

mask参数：掩码变换函数

keypoint参数：关键点变换函数

bbox参数：边框变换函数

always_apply参数：是否这个变换总是应用

p参数：应用此变换的概率，默认为1.0

NoOp (support keypoints) #什么也不做
Transpose (support keypoints)
#转置：交换行与列

p参数:表示应用此变换的概率，默认为0.5

（8）、不支持关键点的变换

CoarseDropout
GridDistortion
GridDropout
MaskDropout
OpticalDistortion
RandomGridShuffle
RandomSizedBBoxSafeCrop
2、Pixel-level transforms (support any additional targets) (像素变换)
Blur
CLAHE
ChannelDropout
ChannelShuffle
ColorJitter
Downscale
Emboss
Equalize
FDA
FancyPCA
FromFloat
GaussNoise
GaussianBlur
GlassBlur
HistogramMatching
HueSaturationValue
ISONoise
ImageCompression
InvertImg
MedianBlur
MotionBlur
MultiplicativeNoise
Normalize
PixelDistributionAdaptation
Posterize
RGBShift
RandomBrightnessContrast
RandomFog
RandomGamma
RandomRain
RandomShadow
RandomSnow
RandomSunFlare
RandomToneCurve
Sharpen
Solarize
Superpixels
TemplateTransform
ToFloat
ToGray
ToSepia
​
