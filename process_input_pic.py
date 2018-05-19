import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================================
# -----------------生成图片路径和标签的List------------------------------------

train_dir = 'D:/train_data/image_data/input_data'

circle = []
label_circle = []
rectangle = []
label_rectangle = []
triangle = []
label_triangle = []


# step1：获取图片路径名，存放到
# 对应的列表中，同时贴上标签，存放到label列表中。
def get_files(file_dir, ratio):
    """
    读入图片，按比例输出训练集和验证集
    :param file_dir: 图片路径名
    :param ratio: 验证集的比例（分数）
    :return: 训练集图片和标签的list，验证集图片和标签的list
    """
    for file in os.listdir(file_dir + '/triangle'):
        triangle.append(file_dir + '/triangle' + '/' + file)
        label_triangle.append(0)
    for file in os.listdir(file_dir + '/circle'):
        circle.append(file_dir + '/circle' + '/' + file)
        label_circle.append(1)
    for file in os.listdir(file_dir + '/rectangle'):
        rectangle.append(file_dir + '/rectangle' + '/' + file)
        label_rectangle.append(2)

    # step2：对生成的图片路径和标签List做打乱处理把cat和dog合起来组成一个list（img和lab）
    image_list = np.hstack((circle, rectangle, triangle))
    label_list = np.hstack((label_circle, label_rectangle, label_triangle))

    # 利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # 从打乱的temp中再取出list（img和lab）
    # image_list = list(temp[:, 0])
    # label_list = list(temp[:, 1])
    # label_list = [int(i) for i in label_list]
    # return image_list, label_list

    # 将所有的img和lab转换成list
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])

    # 将所得List分为两部分，一部分用来训练tra，一部分用来测试val
    # ratio是测试集的比例
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample * ratio))  # 测试样本数
    n_train = n_sample - n_val  # 训练样本数

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    print("训练个数：%d", n_train)
    print("测试个数：%d", n_val)

    return tra_images, tra_labels, val_images, val_labels


# ---------------------------------------------------------------------------
# --------------------生成Batch----------------------------------------------

# step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue，因为img和lab
# 是分开的，所以使用tf.train.slice_input_producer()，然后用tf.read_file()从队列中读取图像
#   image_W, image_H, ：设置好固定的图像高度和宽度
#   设置batch_size：每个batch要放多少张图片
#   capacity：一个队列最大多少
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # 转换类型
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # read img from a queue

    # step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

    # step4：生成batch
    # image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32
    # label_batch: 1D tensor [batch_size], dtype=tf.int32
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)
    # 重新排列label，行数为[batch_size]

    label_batch = tf.reshape(label_batch, [batch_size])

    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch

    # ========================================================================</span>


def list_to_iterator(images, labels):
    '''
    将list转化成iterator
    :param images: 输入的image-list
    :param labels: 输入的labels-list
    :return: 返回（image-iter,labels-iter）
    '''
    images_iter = iter(images)
    labels_iter = iter(labels)
    return images_iter, labels_iter


def get_nparray_batch(images_iter, labels_iter, image_W, image_H, channal, batch_size):
    '''
    得到image&label的batch，用于输入图片及labels
    :param images_iter:
    :param labels_iter:
    :param image_W:
    :param image_H:
    :param batch_size:
    :return:
    '''
    batch_x = np.zeros([batch_size, image_W * image_H * channal])
    batch_y = np.zeros([batch_size])

    def read_img():
        '''
        获取一张图片的内容
        :return:
        '''
        img = Image.open(next(images_iter))
        # plt.imshow(img)
        # plt.show()
        image = np.array(img.resize([64, 64]))
        return image

    for i in range(batch_size):
        image_arr = read_img()
        label_arr = np.array(next(labels_iter))
        batch_x[i, :] = image_arr.flatten() / 255.0
        batch_y[i] = label_arr
    return batch_x, batch_y


if __name__ == '__main__':
    train_dir = 'D:/train_data/image_data/input_data'  # 训练样本的读入路径
    logs_train_ckpt_dir = 'D:/train_data/image_data/input_data/ckpt'  # logs存储路径
    logs_train_pb_dir = 'D:/train_data/image_data/input_data/pb/output.pb'

    train, train_label, val, val_label = get_files(train_dir, 0.2)
    train_iter, train_label_iter = list_to_iterator(train, train_label)

    image_batch, label_batch = get_nparray_batch(
        train_iter, train_label_iter, 64, 64, 3, 1)
    print("image_batch", image_batch, "label_batch", label_batch)
    image_batch, label_batch = get_nparray_batch(
        train_iter, train_label_iter, 64, 64, 3, 3)
    print("image_batch", image_batch, "label_batch", label_batch)
    image_batch, label_batch = get_nparray_batch(
        train_iter, train_label_iter, 64, 64, 3, 1)
    print("image_batch", image_batch, "label_batch", label_batch)
    image_batch, label_batch = get_nparray_batch(
        train_iter, train_label_iter, 64, 64, 3, 1)
    print("image_batch", image_batch, "label_batch", label_batch)
