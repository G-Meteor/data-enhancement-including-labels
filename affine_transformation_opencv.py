import cv2
import os
import math
import numpy as np
#from PIL import Image
from cv2.cv2 import INTER_LINEAR
import pandas as pd


def mkdir_folder(path):
    if os.path.exists(path) == 1:
        print(path + '文件夹已存在')
    else:
        os.mkdir(path)


def rotateDatas(angle, new_img_folder, new_gt_folder, new_img_punc_folder):
    gt_folder = os.listdir(src_gt_folder)
    for i in range(0, len(gt_folder)):
        gt_file = src_gt_folder + gt_folder[i]
        img_file = src_img_folder + gt_folder[i]
        gt_name = os.listdir(gt_file)
        img_name = os.listdir(img_file)

        # 创建新目录
        new_img_folder2 = new_img_folder + "/rotate_" + str(angle)
        mkdir_folder(new_img_folder2)
        new_img_folder2 = new_img_folder2 + "/" + gt_folder[i]
        mkdir_folder(new_img_folder2)
        new_gt_folder2 = new_gt_folder + "/rotate_" + str(angle)
        mkdir_folder(new_gt_folder2)
        new_gt_folder2 = new_gt_folder2 + "/" + gt_folder[i]
        mkdir_folder(new_gt_folder2)
        new_img_punc_folder2 = new_img_punc_folder + "/rotate_" + str(angle)
        mkdir_folder(new_img_punc_folder2)
        new_img_punc_folder2 = new_img_punc_folder2 + "/" + gt_folder[i]
        mkdir_folder(new_img_punc_folder2)

        for j in range(0, len(img_name)):
            img_path = img_file + "/" + img_name[j]
            gt_path = gt_file + "/" + gt_name[j]
            # print(img_path)
            rotateOneData(img_path, gt_path, angle, new_img_folder2,
                          new_gt_folder2, new_img_punc_folder2)
    print("**********************旋转任务完成*************************")


def panDatas(distance, new_img_folder, new_gt_folder, new_img_punc_folder):
    gt_folder = os.listdir(src_gt_folder)
    for i in range(0, len(gt_folder)):
        gt_file = src_gt_folder + gt_folder[i]
        img_file = src_img_folder + gt_folder[i]
        gt_name = os.listdir(gt_file)
        img_name = os.listdir(img_file)
        # print(str(distance))
        # 创建新目录
        new_img_folder2 = new_img_folder + "/pan_" + str(distance)
        mkdir_folder(new_img_folder2)
        new_img_folder2 = new_img_folder2 + "/" + gt_folder[i]
        mkdir_folder(new_img_folder2)
        new_gt_folder2 = new_gt_folder + "/pan_" + str(distance)
        mkdir_folder(new_gt_folder2)
        new_gt_folder2 = new_gt_folder2 + "/" + gt_folder[i]
        mkdir_folder(new_gt_folder2)
        new_img_punc_folder2 = new_img_punc_folder + "/pan_" + str(distance)
        mkdir_folder(new_img_punc_folder2)
        new_img_punc_folder2 = new_img_punc_folder2 + "/" + gt_folder[i]
        mkdir_folder(new_img_punc_folder2)

        for j in range(0, len(img_name)):
            img_path = img_file + "/" + img_name[j]
            gt_path = gt_file + "/" + gt_name[j]
            # print(img_path)
            panOneData(img_path, gt_path, distance, new_img_folder2,
                       new_gt_folder2, new_img_punc_folder2)
    print("**********************旋转任务完成*************************")


def zoomDatas(scale, new_img_folder, new_gt_folder, new_img_punc_folder):
    gt_folder = os.listdir(src_gt_folder)
    for i in range(0, len(gt_folder)):
        gt_file = src_gt_folder + gt_folder[i]
        img_file = src_img_folder + gt_folder[i]
        gt_name = os.listdir(gt_file)
        img_name = os.listdir(img_file)

        # 创建新目录
        new_img_folder2 = new_img_folder + "/zoom_" + str(scale)
        mkdir_folder(new_img_folder2)
        new_img_folder2 = new_img_folder2 + "/" + gt_folder[i]
        mkdir_folder(new_img_folder2)
        new_gt_folder2 = new_gt_folder + "/zoom_" + str(scale)
        mkdir_folder(new_gt_folder2)
        new_gt_folder2 = new_gt_folder2 + "/" + gt_folder[i]
        mkdir_folder(new_gt_folder2)
        new_img_punc_folder2 = new_img_punc_folder + "/zoom_" + str(scale)
        mkdir_folder(new_img_punc_folder2)
        new_img_punc_folder2 = new_img_punc_folder2 + "/" + gt_folder[i]
        mkdir_folder(new_img_punc_folder2)

        for j in range(0, len(img_name)):
            img_path = img_file + "/" + img_name[j]
            gt_path = gt_file + "/" + gt_name[j]
            # print(img_path)
            zoomOneData(img_path, gt_path, scale, new_img_folder2,
                        new_gt_folder2, new_img_punc_folder2)
    print("**********************缩放任务完成*************************")


def affineDatas(new_img_folder, new_gt_folder, new_img_punc_folder):
    gt_folder = os.listdir(src_gt_folder)
    for i in range(0, len(gt_folder)):
        gt_file = src_gt_folder + gt_folder[i]
        img_file = src_img_folder + gt_folder[i]
        gt_name = os.listdir(gt_file)
        img_name = os.listdir(img_file)

        # 创建新目录
        new_img_folder2 = new_img_folder + "/affine"
        mkdir_folder(new_img_folder2)
        new_img_folder2 = new_img_folder2 + "/" + gt_folder[i]
        mkdir_folder(new_img_folder2)
        new_gt_folder2 = new_gt_folder + "/affine"
        mkdir_folder(new_gt_folder2)
        new_gt_folder2 = new_gt_folder2 + "/" + gt_folder[i]
        mkdir_folder(new_gt_folder2)
        new_img_punc_folder2 = new_img_punc_folder + "/affine"
        mkdir_folder(new_img_punc_folder2)
        new_img_punc_folder2 = new_img_punc_folder2 + "/" + gt_folder[i]
        mkdir_folder(new_img_punc_folder2)

        for j in range(0, len(img_name)):
            img_path = img_file + "/" + img_name[j]
            gt_path = gt_file + "/" + gt_name[j]
            # print(img_path)
            affineOneData(img_path, gt_path, new_img_folder2,
                          new_gt_folder2, new_img_punc_folder2)
    print("**********************仿射任务完成*************************")

# flipCode – Flag to specify how to flip the array. 0 means flipping around the x-axis.
# Positive value (for example, 1) means flipping around y-axis.
# Negative value (for example, -1) means flipping around both axes. See the discussion below for the formulas.
def flipDatas(flipCode,new_img_folder, new_gt_folder, new_img_punc_folder):
    gt_folder = os.listdir(src_gt_folder)
    for i in range(0, len(gt_folder)):
        gt_file = src_gt_folder + gt_folder[i]
        img_file = src_img_folder + gt_folder[i]
        gt_name = os.listdir(gt_file)
        img_name = os.listdir(img_file)

        # 创建新目录
        new_img_folder2 = new_img_folder + "/flip_"+str(flipCode)
        mkdir_folder(new_img_folder2)
        new_img_folder2 = new_img_folder2 + "/" + gt_folder[i]
        mkdir_folder(new_img_folder2)
        new_gt_folder2 = new_gt_folder + "/flip_"+str(flipCode)
        mkdir_folder(new_gt_folder2)
        new_gt_folder2 = new_gt_folder2 + "/" + gt_folder[i]
        mkdir_folder(new_gt_folder2)
        new_img_punc_folder2 = new_img_punc_folder + "/flip_"+str(flipCode)
        mkdir_folder(new_img_punc_folder2)
        new_img_punc_folder2 = new_img_punc_folder2 + "/" + gt_folder[i]
        mkdir_folder(new_img_punc_folder2)

        for j in range(0, len(img_name)):
            img_path = img_file + "/" + img_name[j]
            gt_path = gt_file + "/" + gt_name[j]
            # print(img_path)
            flipOneData(img_path, gt_path, flipCode,new_img_folder2,
                          new_gt_folder2, new_img_punc_folder2)
    print("**********************翻转任务完成*************************")

def rotateOneData(img_path, gt_path, angle, new_img_folder, new_gt_folder, new_img_punc_folder):
    center = []
    radian = angle * math.pi / 180
    distance = [0, 0]
    scale = 1
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_shape = img.shape
    center.append(img_shape[1] / 2)
    center.append(img_shape[0] / 2)

    # 处理图片
    img_rot = ImageRotate(img, center, angle)
    img_copy = img_rot.copy()
    # 处理一张图的所有点
    img_name, new_lines = processAllPoints(
        gt_path, center, radian, distance, scale, img_copy)
    # 保存
    new_gt_name = new_gt_folder + "/" + img_name.replace("png\n", "txt")
    with open(new_gt_name, "wt") as out_file:
        # print(str(new_lines))
        out_file.writelines(new_lines)
    new_img_name = new_img_folder + "/" + img_name.replace("\n", "")
    cv2.imwrite(new_img_name, img_rot)
    new_img_punc_name = new_img_punc_folder + "/" + img_name.replace("\n", "")
    cv2.imwrite(new_img_punc_name, img_copy)
    print(img_name.replace("\n", "") + ":已保存———————————————")


def panOneData(img_path, gt_path, distance, new_img_folder, new_gt_folder, new_img_punc_folder):
    center = []
    angle = 0
    radian = angle * math.pi / 180
    #distance = [0, 0]
    scale = 1
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_shape = img.shape
    center.append(img_shape[1] / 2)
    center.append(img_shape[0] / 2)

    # 处理图片
    img_rot = ImagePanning(img,  distance)
    img_copy = img_rot.copy()
    # 处理一张图的所有点
    img_name, new_lines = processAllPoints(
        gt_path, center, radian, distance, scale, img_copy)
    # 保存
    new_gt_name = new_gt_folder + "/" + img_name.replace("png\n", "txt")
    with open(new_gt_name, "wt") as out_file:
        # print(str(new_lines))
        out_file.writelines(new_lines)
    new_img_name = new_img_folder + "/" + img_name.replace("\n", "")
    cv2.imwrite(new_img_name, img_rot)
    new_img_punc_name = new_img_punc_folder + "/" + img_name.replace("\n", "")
    cv2.imwrite(new_img_punc_name, img_copy)
    print(img_name.replace("\n", "") + ":已保存———————————————")


def zoomOneData(img_path, gt_path, scale, new_img_folder, new_gt_folder, new_img_punc_folder):
    center = []
    angle = 0
    radian = angle * math.pi / 180
    distance = [0, 0]
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_shape = img.shape
    center.append(img_shape[1] / 2)
    center.append(img_shape[0] / 2)

    # 处理图片
    img_rot = ImageZoom(img, center, scale)
    img_copy = img_rot.copy()
    # 处理一张图的所有点
    img_name, new_lines = processAllPoints(
        gt_path, center, radian, distance, scale, img_copy)
    # 保存
    new_gt_name = new_gt_folder + "/" + img_name.replace("png\n", "txt")
    with open(new_gt_name, "wt") as out_file:
        # print(str(new_lines))
        out_file.writelines(new_lines)
    new_img_name = new_img_folder + "/" + img_name.replace("\n", "")
    cv2.imwrite(new_img_name, img_rot)
    new_img_punc_name = new_img_punc_folder + "/" + img_name.replace("\n", "")
    cv2.imwrite(new_img_punc_name, img_copy)
    print(img_name.replace("\n", "") + ":已保存———————————————")


def affineOneData(img_path, gt_path, new_img_folder, new_gt_folder, new_img_punc_folder):
    srcTri = []
    dstTri = []
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_shape = img.shape
    srcTri.append([0, 0])
    srcTri.append([img_shape[1] - 1, 0])
    srcTri.append([0, img_shape[0] - 1])
    srcTri = np.float32(srcTri)
    # print(srcTri)
    dstTri.append([img_shape[1] * 0.0, img_shape[0] * 0.33])
    dstTri.append([img_shape[1] * 0.85, img_shape[0] * 0.25])
    dstTri.append([img_shape[1] * 0.15, img_shape[0] * 0.7])
    dstTri = np.float32(dstTri)
    warp_mat = cv2.getAffineTransform(srcTri, dstTri)

    # 处理图片
    img_rot = ImageAffine(img, warp_mat)
    img_copy = img_rot.copy()
    # 处理一张图的所有点
    img_name, new_lines = processAllPoints2(gt_path, warp_mat, img_copy)
    # 保存
    new_gt_name = new_gt_folder + "/" + img_name.replace("png\n", "txt")
    with open(new_gt_name, "wt") as out_file:
        # print(str(new_lines))
        out_file.writelines(new_lines)
    new_img_name = new_img_folder + "/" + img_name.replace("\n", "")
    cv2.imwrite(new_img_name, img_rot)
    new_img_punc_name = new_img_punc_folder + "/" + img_name.replace("\n", "")
    cv2.imwrite(new_img_punc_name, img_copy)
    print(img_name.replace("\n", "") + ":已保存———————————————")


def flipOneData(img_path, gt_path, flipCode, new_img_folder, new_gt_folder, new_img_punc_folder):

    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_shape = img.shape
    # 处理图片
    img_rot = ImageFlip(img, flipCode)
    img_copy = img_rot.copy()
    # 处理一张图的所有点
    img_name, new_lines = processAllPoints3(
        gt_path, flipCode, img_copy, img_shape)
    # 保存
    new_gt_name = new_gt_folder + "/" + img_name.replace("png\n", "txt")
    with open(new_gt_name, "wt") as out_file:
        # print(str(new_lines))
        out_file.writelines(new_lines)
    new_img_name = new_img_folder + "/" + img_name.replace("\n", "")
    cv2.imwrite(new_img_name, img_rot)
    new_img_punc_name = new_img_punc_folder + "/" + img_name.replace("\n", "")
    cv2.imwrite(new_img_punc_name, img_copy)
    print(img_name.replace("\n", "") + ":已保存———————————————")

# 图像仿射变换，旋转、平移、缩放、仿射


def ImageRotate(src, center, angle):
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_shape = src.shape
    dst = cv2.warpAffine(
        src, M, (img_shape[1], img_shape[0]), borderValue=(255, 255, 255))
    # flags=CV_INTER_LINEAR
    # cv2.imwrite('test_wAff.png', dst)
    return dst




def ImageFlip(src, flipCode):
    dst = cv2.flip(src, flipCode)
    return dst


def ImagePanning(src, distance):
    x = distance[0]
    y = distance[1]
    M = [[1, 0, x], [0, 1, y]]
    M = np.float32(M)
    img_shape = src.shape
    dst = cv2.warpAffine(
        src, M, (img_shape[1], img_shape[0]), borderValue=(255, 255, 255))
    return dst


def ImageZoom(src, center, scale):
    M = cv2.getRotationMatrix2D(center, 0, scale)
    img_shape = src.shape
    dst = cv2.warpAffine(
        src, M, (img_shape[1], img_shape[0]), borderValue=(255, 255, 255))
    return dst


def ImageAffine(src, M):
    img_shape = src.shape
    dst = cv2.warpAffine(
        src, M, (img_shape[1], img_shape[0]), borderValue=(255, 255, 255))
    return dst


def processAllPoints(gt_path, center, radian, distance, scale, img_copy):
    with open(gt_path, encoding='utf-8') as f:
        lines = f.readlines()
        new_lines = lines[:]
        img_name = lines[0]
        for i in range(0, 251):
            point = []
            point.append(float(lines[i * 3 + 2]))
            point.append(float(lines[i * 3 + 3]))
            point_aff = getPointAffinedPos(
                point, center, radian, distance, scale)
            new_lines[i * 3 + 2] = str(point_aff[0]) + '\n'
            new_lines[i * 3 + 3] = str(point_aff[1]) + '\n'
            cv2.circle(img_copy, [round(point_aff[0]),
                       round(point_aff[1])], 5, (0, 0, 255), -1)
    return img_name, new_lines


def processAllPoints2(gt_path, M, img_copy):
    with open(gt_path, encoding='utf-8') as f:
        lines = f.readlines()
        new_lines = lines[:]
        img_name = lines[0]
        for i in range(0, 251):
            point = []
            point.append(float(lines[i * 3 + 2]))
            point.append(float(lines[i * 3 + 3]))
            point_aff = getPointAffinedPos2(point, M)
            new_lines[i * 3 + 2] = str(point_aff[0]) + '\n'
            new_lines[i * 3 + 3] = str(point_aff[1]) + '\n'
            cv2.circle(img_copy, [round(point_aff[0]),
                       round(point_aff[1])], 5, (0, 0, 255), -1)
    return img_name, new_lines


def processAllPoints3(gt_path, flipCode, img_copy, img_shape):
    new_id = pd.read_excel("./翻转处理后人脸关键点对应关系.xls", usecols=[1])
    with open(gt_path, encoding='utf-8') as f:
        lines = f.readlines()
        new_lines = lines[:]
        img_name = lines[0]
        for i in range(0, 251):
            point = []
            point.append(float(lines[i * 3 + 2]))
            point.append(float(lines[i * 3 + 3]))
            # print(point)
            #point = np.float32(point)
            point_aff = getPointFlipPos(
                point, [img_shape[0], img_shape[1]], flipCode)
            new_i = i
            if flipCode >= 0:
                new_i = new_id.values[i][0]
            new_lines[new_i * 3 + 2] = str(point_aff[0]) + '\n'
            new_lines[new_i * 3 + 3] = str(point_aff[1]) + '\n'
            new_lines[new_i * 3 + 4] = lines[i * 3 + 4]
            cv2.putText(img_copy, str(new_i), [round(point_aff[0]),
                        round(point_aff[1])], cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 0, 255), 1)
            # cv2.circle(img_copy, [round(point_aff[0]),
            #            round(point_aff[1])], 5, (0, 0, 255), -1)
    return img_name, new_lines

# 获取指定像素点放射变换后的新的坐标位置


def getPointAffinedPos(src, center, radian, distance, scale):
    dst = []
    x = src[0] - center[0]
    y = src[1] - center[1]
    dst.append((x * math.cos(radian) + y * math.sin(radian))
               * scale + distance[0] + center[0])
    dst.append((-x * math.sin(radian) + y * math.cos(radian))
               * scale + distance[1] + center[1])
    return dst


def getPointAffinedPos2(src, M):
    dst = []
    x = M[0][0] * src[0] + M[0][1] * src[1] + M[0][2]
    y = M[1][0] * src[0] + M[1][1] * src[1] + M[1][2]
    dst.append(x)
    dst.append(y)
    return dst


def getPointFlipPos(src, size, flipCode):
    if flipCode == 0:
        src[1] = size[0] - src[1]
    elif flipCode > 0:
        src[0] = size[1] - src[0]
    else:
        src[0] = size[1] - src[0]
        src[1] = size[0] - src[1]
    return src


if __name__ == "__main__":
    src_img_folder = "./21testyasuo/"
    src_gt_folder = "./21test_yasuo_251gt_smooth/"
    new_img_folder = "./new_image"
    new_gt_folder = "./new_ground_truth"
    new_img_punc_folder = "./new_image_punctuation"
    mkdir_folder(new_img_folder)
    mkdir_folder(new_gt_folder)
    mkdir_folder(new_img_punc_folder)

    angle = 30
    # rotateDatas(angle, new_img_folder, new_gt_folder, new_img_punc_folder)
    # affineDatas(new_img_folder, new_gt_folder, new_img_punc_folder)
    scale = 0.8
    #zoomDatas(scale, new_img_folder, new_gt_folder, new_img_punc_folder)
    distance = [100, 100]
    #panDatas(distance, new_img_folder, new_gt_folder, new_img_punc_folder)
    flipCode = -1 
    flipDatas(flipCode,new_img_folder, new_gt_folder, new_img_punc_folder)
    print("**********************所有任务完成*************************")
