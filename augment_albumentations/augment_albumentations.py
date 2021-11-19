# coding=utf-8
import cv2
import albumentations as A
import os
import numpy as np
import queue
import threading
import time
import xlwt

THREAD_NUM = 1
queueLock = threading.Lock()
exitFlag = 0
KEYPOINT_COLOR = (0, 0, 255)

# 处理队列中数据,workQueue参数为数据队列，operate参数为要进行的操作（支持"image"与"image_keypoints"两个参数）


def consumer(workQueue, operate):
    while not exitFlag:
        queueLock.acquire()
        if not workQueue.empty():
            print(threading.currentThread())
            # 功能句
            if(operate == "image"):
                img_path, new_img_path, process_name = workQueue.get()
                process_oneimage(img_path, new_img_path, process_name)
            elif(operate == "image_keypoints"):
                img_path, new_img_path, gt_path, new_gt_path, process_name = workQueue.get()
                process_oneimage_keypoints_xy(
                    img_path, new_img_path, gt_path, new_gt_path, process_name)
            print()
            # print("Thread-%s processing %s" % (threadID, data))
        queueLock.release()
        time.sleep(0)


# 创建线程
def createThreads(target, args):
    threads = []
    for i in range(0, THREAD_NUM):
        # threadID = i+1
        # 创建新线程
        # thread = threading.Thread(target=consumer, args=(workQueue,))
        thread = threading.Thread(target=target, args=args)
        # print(thread.getName())
        thread.start()
        threads.append(thread)
    return threads

# 等待所有任务处理完成


def waitTask2End(workQueue, threads):
    # 等待队列清空
    while not workQueue.empty():
        pass
    # 通知线程是时候退出
    global exitFlag
    exitFlag = 1
    # 等待所有线程完成
    for t in threads:
        t.join()
    print("退出主线程")

# 创建目录


def mkdir_folder(path):
    if os.path.exists(path) == 1:
        print(path + '文件夹已存在')
    else:
        os.mkdir(path)

# 支持中文路径的图片读取


def cv2_imread(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    return img

# 支持中文路径的图片写入


def cv2_imwrite(path, img):
    format = '.'+path.split(".")[-1]
    cv2.imencode(format, img)[1].tofile(path)

# 在图片上可视化关键点


def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=7):
    image = image.copy()
    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, color, -1)
    cv2.imwrite("test_albu.png", image)


# 对图片进行随机仿射变换
def affine_image(image):
    transform = A.Compose([A.Affine(p=1, shear=[-45, 45])], )
    transformed = transform(image=image)
    return transformed['image']

# 对图片+关键点进行随机仿射变换


def affine_image_keypoints_xy(image, keypoints):
    transform = A.Compose(
        [A.Affine(p=1, shear=[-45, 45])],
        keypoint_params=A.KeypointParams(format='xy'),
    )
    transformed = transform(image=image, keypoints=keypoints)
    return transformed['image'], transformed['keypoints']


# 对图片进行随机剪裁
def crop_image(image):
    height = image.shape[0]
    width = image.shape[1]
    transform = A.Compose(
        [A.RandomResizedCrop(p=1, height=height, width=width)], )
    transformed = transform(image=image)
    return transformed['image']


# 对图片+关键点进行随机剪裁
def crop_image_keypoints_xy(image, keypoints):
    height = image.shape[0]
    width = image.shape[1]
    transform = A.Compose(
        [A.RandomResizedCrop(p=1, height=height, width=width)],
        keypoint_params=A.KeypointParams(format='xy'),
    )
    transformed = transform(image=image, keypoints=keypoints)
    return transformed['image'], transformed['keypoints']


# 对图片进行随机翻转,包括水平翻转、垂直翻转、水平垂直同时进行翻转
def flip_image(image):
    transform = A.Compose([A.Flip(p=1)], )
    transformed = transform(image=image)
    return transformed['image']


# 带关键点的随机翻转，仅支持关键点没有顺序的情况
def flip_image_keypoints_xy(image, keypoints):
    transform = A.Compose(
        [A.Flip(p=1)],
        keypoint_params=A.KeypointParams(format='xy'),
    )
    transformed = transform(image=image, keypoints=keypoints)
    return transformed['image'], transformed['keypoints']


# 对图片进行随机旋转
def rotate_image(image):
    transform = A.Compose([A.Affine(p=1, rotate=[-60, 60])], )
    transformed = transform(image=image)
    return transformed['image']

# 对图片+关键点进行随机旋转


def rotate_image_keypoints_xy(image, keypoints):
    transform = A.Compose(
        [A.Affine(p=1, rotate=[-60, 60])],
        keypoint_params=A.KeypointParams(format='xy'),
    )
    transformed = transform(image=image, keypoints=keypoints)
    return transformed['image'], transformed['keypoints']


# 对图片进行随机缩放
def scale_image(image):
    transform = A.Compose([A.Affine(p=1, scale=[0.5, 1.5])])
    transformed = transform(image=image)
    return transformed['image']


# 对图片+关键点进行随机缩放
def scale_image_keypoints_xy(image, keypoints):
    transform = A.Compose(
        [A.Affine(p=1, scale=[0.5, 1.5])],
        keypoint_params=A.KeypointParams(format='xy'),
    )
    transformed = transform(image=image, keypoints=keypoints)
    return transformed['image'], transformed['keypoints']


# 读取关键点数据
def readPoints(path):
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        num = int(lines[1])
        points = []
        for i in range(0, num):
            point = []
            point.append(float(lines[i * 3 + 2]))
            point.append(float(lines[i * 3 + 3]))
            points.append(point)
    return points


# 写入关键点数据
def writePoints(path, points, raw_lines):
    new_lines = raw_lines[:]
    for i in range(0, int(len(points)/2)):
        new_lines[i * 3 + 2] = str(points[i*2]) + '\n'
        new_lines[i * 3 + 3] = str(points[i*2+1]) + '\n'
    with open(path, "wt") as out_file:
        out_file.writelines(new_lines)


# 处理一张图片数据
def process_oneimage(img_path, new_img_path, process_name):
    img = cv2_imread(img_path)
    if(process_name == "affine"):
        new_img = affine_image(img)
        cv2_imwrite(new_img_path, new_img)
        print("******正在进行仿射数据增强******")
    elif(process_name == "crop"):
        new_img = crop_image(img)
        cv2_imwrite(new_img_path, new_img)
        print("******正在进行剪裁数据增强******")
    elif(process_name == "flip"):
        new_img = flip_image(img)
        cv2_imwrite(new_img_path, new_img)
        print("******正在进行翻转数据增强******")
    elif(process_name == "rotate"):
        new_img = rotate_image(img)
        cv2_imwrite(new_img_path, new_img)
        print("******正在进行旋转数据增强******")
    elif(process_name == "scale"):
        new_img = scale_image(img)
        cv2_imwrite(new_img_path, new_img)
        print("******正在进行缩放数据增强******")
    else:
        print("******无效操作******")

# 处理一张图片数据(包括关键点坐标)


def process_oneimage_keypoints_xy(img_path, new_img_path, gt_path, new_gt_path, process_name):
    img = cv2_imread(img_path)
    keypoints = readPoints(gt_path)
    with open(gt_path, encoding='utf-8') as f:
        raw_lines = f.readlines()
    if(process_name == "affine"):
        new_img, new_keypoints = affine_image_keypoints_xy(
            img, keypoints)
        #vis_keypoints(new_img, new_keypoints)
        cv2_imwrite(new_img_path, new_img)
        writePoints(new_gt_path, new_keypoints, raw_lines)
        print("******正在进行仿射数据增强******")
    elif(process_name == "crop"):
        new_img, new_keypoints = crop_image_keypoints_xy(
            img, keypoints)
        #vis_keypoints(new_img, new_keypoints)
        cv2_imwrite(new_img_path, new_img)
        writePoints(new_gt_path, new_keypoints, raw_lines)
        print("******正在进行剪裁数据增强******")
    elif(process_name == "flip"):
        new_img, new_keypoints = flip_image_keypoints_xy(
            img, keypoints)
        #vis_keypoints(new_img, new_keypoints)
        cv2_imwrite(new_img_path, new_img)
        writePoints(new_gt_path, new_keypoints, raw_lines)
        print("******正在进行翻转数据增强******")
    elif(process_name == "rotate"):
        new_img, new_keypoints = rotate_image_keypoints_xy(
            img, keypoints)
        #vis_keypoints(new_img, new_keypoints)
        cv2_imwrite(new_img_path, new_img)
        writePoints(new_gt_path, new_keypoints, raw_lines)
        print("******正在进行旋转数据增强******")
    elif(process_name == "scale"):
        new_img, new_keypoints = scale_image_keypoints_xy(
            img, keypoints)
        #vis_keypoints(new_img, new_keypoints)
        cv2_imwrite(new_img_path, new_img)
        writePoints(new_gt_path, new_keypoints, raw_lines)
        print("******正在进行缩放数据增强******")
    else:
        print("******无效操作******")
# ********flask文件上传************
# 批处理图片数据


def process_images(src_img_folder, process_name):
    workQueue = queue.Queue()  # 用于线程处理的数据队列
    threads = createThreads(target=consumer, args=(workQueue, "image"))
    img_folder = os.listdir(src_img_folder)
    new_img_folder = src_img_folder + "_"+process_name
    mkdir_folder(new_img_folder)
    for i in range(0, len(img_folder)):
        img_file = src_img_folder + "/" + img_folder[i]
        img_name = os.listdir(img_file)
        # print(img_file)
        # 创建目录
        new_img_folder2 = new_img_folder + "/" + img_folder[i]
        mkdir_folder(new_img_folder2)
        for j in range(0, len(img_name)):
            img_path = img_file + "/" + img_name[j]
            new_img_path = new_img_folder2 + "/" + img_name[j]
            queueLock.acquire()
            workQueue.put([img_path, new_img_path, process_name])
            queueLock.release()
            # process_oneimage(img_path,new_img_path,process_name)
    waitTask2End(workQueue, threads)
    print("**********************全部任务完成*************************")


# 批处理图片+关键点数据
def process_images_keypoints_xy(src_img_folder, src_gt_folder, process_name):
    workQueue = queue.Queue()  # 用于线程处理的数据队列
    threads = createThreads(
        target=consumer, args=(workQueue, "image_keypoints"))
    img_folder = os.listdir(src_img_folder)
    new_img_folder = src_img_folder + "_"+process_name
    mkdir_folder(new_img_folder)
    gt_folder = os.listdir(src_gt_folder)
    new_gt_folder = src_gt_folder + "_"+process_name
    mkdir_folder(new_gt_folder)
    for i in range(0, len(img_folder)):
        img_file = src_img_folder + "/" + img_folder[i]
        img_name = os.listdir(img_file)
        gt_file = src_gt_folder + "/" + gt_folder[i]
        gt_name = os.listdir(gt_file)
        # print(img_file)
        # 创建目录
        new_img_folder2 = new_img_folder + "/" + img_folder[i]
        mkdir_folder(new_img_folder2)
        new_gt_folder2 = new_gt_folder + "/" + gt_folder[i]
        mkdir_folder(new_gt_folder2)

        for j in range(0, len(img_name)):
            img_path = img_file + "/" + img_name[j]
            new_img_path = new_img_folder2 + "/" + img_name[j]
            gt_path = gt_file + "/" + gt_name[j]
            new_gt_path = new_gt_folder2 + "/" + gt_name[j]
            # print(img_path)
            #process_oneimage_keypoints_xy(img_path, new_img_path, gt_path, new_gt_path, process_name)
            queueLock.acquire()
            workQueue.put([img_path, new_img_path, gt_path,
                          new_gt_path, process_name])
            queueLock.release()
    waitTask2End(workQueue, threads)
    print("**********************全部任务完成*************************")

# 自定义写入规则函数


def write_excel(book, sheet, book_name):
    root_path = os.path.dirname(os.path.dirname(__file__))
    img_folder = root_path+"/测试数据/images"
    gt_folder = root_path+"/测试数据/ground_truth"

    operations = ["affine", "crop", "flip", "rotate", "scale"]
    for i in range(0, len(operations)):
        sheet.write(i+1, 0, operations[i]+"_time")
    global exitFlag
    sheet.write(0, 0, "thread_num")
    i = 1
    global THREAD_NUM
    while THREAD_NUM <= 101:
        sheet.write(0, i, THREAD_NUM)
        # for j in range(0, len(operations)):
        j = 0
        start = time.clock()
        # print(operate)
        exitFlag = 0
        process_images_keypoints_xy(img_folder, gt_folder, operations[j])
        end = time.clock()
        sheet.write(j+1, i, end-start)
        print(end-start)
        book.save(book_name)
        print(book_name+"_线程数"+str(THREAD_NUM)+":创建完成**************")
        THREAD_NUM += 5
        i += 1


def creat_excel(book_name, sheet_name):
    book = xlwt.Workbook(encoding='utf-8')  # 创建Workbook，相当于创建Excel
    # 创建sheet，Sheet1为表的名字，cell_overwrite_ok为是否覆盖单元格
    sheet = book.add_sheet(sheet_name, cell_overwrite_ok=True)
    write_excel(book, sheet, book_name)  # 自定义写入规则
    book.save(book_name)
    print(book_name+"创建完成**************")


def test():
    book_name = "times_process_images_keyponts_300.xls"
    sheet_name = 'Sheet1'
    creat_excel(book_name, sheet_name)
    # root_path = os.path.dirname(os.path.dirname(__file__))
    # img_folder = root_path+"/测试数据/images"
    # gt_folder = root_path+"/测试数据/ground_truth"

    # operations = ["affine", "crop", "flip", "rotate", "scale"]

    # global exitFlag
    # for operate in operations:
    #     start = time.clock()
    #     #print(operate)
    #     exitFlag = 0
    #     process_images(img_folder, operate)
    #     end = time.clock()
    #     print(end-start)

    # 处理所有图片数据,支持操作有affine、crop、flip、rotate、scale

    # exitFlag = 0
    # process_images(img_folder, "affine")
    # exitFlag = 0
    # process_images(img_folder, "crop")
    # exitFlag = 0
    # process_images(img_folder, "flip")
    # process_images(img_folder, "rotate")
    # process_images(img_folder, "scale")

    # 处理所有图片+关键点数据,支持操作有affine、crop、flip、rotate、scale
    # process_images_keypoints_xy(img_folder, gt_folder, "affine")
    # process_images_keypoints_xy(img_folder, gt_folder, "crop")
    # process_images_keypoints_xy(img_folder, gt_folder, "flip")
    # process_images_keypoints_xy(img_folder, gt_folder, "rotate")
    # process_images_keypoints_xy(img_folder, gt_folder, "scale")

    # ---------------------------------------------------
    # img_test_path = root_path+'/augment_albumentations/test.png'
    # image = cv2_imread(img_test_path)
    # # 接收图片为BGR格式的，若输入图片为RGB图像则需要以下操作
    # # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(image.shape)
    # # 关键点格式为“xy”格式
    # keypoints = [
    #     (167.9536, 878.3683),
    #     (170.3921, 920.2455),
    #     (169.6376, 964.3743),
    #     (751.2452, 1310.0646),
    # ]
    # vis_keypoints(image, keypoints)

    # 只有图片(单张图片)
    # new_image = scale_image(image)
    # print(new_image.shape)
    # cv2.imwrite("test_albu.png", new_image)

    # 图片+关键点
    # new_image, new_keypoints = scale_image_keypoints_xy(image, keypoints)
    # print(new_image.shape)
    # vis_keypoints(new_image, new_keypoints)


if __name__ == "__main__":
    test()  # 用于测试的函数

    root_path = os.path.dirname(os.path.dirname(__file__))
    img_folder = root_path+"/原数据/21testyasuo"
    # 处理所有图片数据,支持操作有affine、crop、flip、rotate、scale
    # process_images(img_folder, "affine")
    # process_images(img_folder, "crop")
    # process_images(img_folder, "flip")
    # process_images(img_folder, "rotate")
    # process_images(img_folder, "scale")

    # 处理所有图片+关键点数据,支持操作有affine、crop、flip、rotate、scale
    gt_folder = root_path+"/原数据/21test_yasuo_251gt_smooth"
    # process_images_keypoints_xy(img_folder, gt_folder, "affine")
    # process_images_keypoints_xy(img_folder, gt_folder, "crop")
    # process_images_keypoints_xy(img_folder, gt_folder, "flip")
    # process_images_keypoints_xy(img_folder, gt_folder, "rotate")
    # process_images_keypoints_xy(img_folder, gt_folder, "scale")
