import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A

# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
KEYPOINT_COLOR = (0, 0 ,255) 

def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=7):
    image = image.copy()
    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, color, -1)
    cv2.imwrite("test_albu.png", image)  

image = cv2.imread('./test.png')
print(image.shape)
keypoints = [
    (167.9536, 878.3683),
    (170.3921, 920.2455),
    (169.6376, 964.3743),
    (751.2452,1310.0646),
]
#vis_keypoints(image, keypoints)

transform = A.Compose(
    [#A. Perspective(scale=[0.5,0.5],keep_size=True,fit_output=True,p=1)
    A.PiecewiseAffine(p=1)], 
    keypoint_params=A.KeypointParams(format='xy')
)
transformed = transform(image=image, keypoints=keypoints)

vis_keypoints(transformed['image'], transformed['keypoints'])
print(transformed['image'].shape)
