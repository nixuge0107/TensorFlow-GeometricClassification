import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from build_train_model import inference


# =======================================================================
# 获取一张图片
def get_one_image(train):
    # 输入参数：train,训练图片的路径
    # 返回参数：image，从训练图片中随机抽取一张图片
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]  # 随机选择测试的图片

    img = Image.open(img_dir)
    plt.imshow(img)
    # plt.show()
    imag = img.resize([64, 64])  # 由于图片在预处理阶段以及resize，因此该命令可略
    image = np.array(imag)
    return image


def get_image(image_path):
    img = Image.open(image_path)
    plt.imshow(img)
    # plt.show()
    imag = img.resize([64, 64])  # 由于图片在预处理阶段以及resize，因此该命令可略
    image = np.array(imag)
    return image


# --------------------------------------------------------------------
# 测试图片
def evaluate_one_image(image_array):
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 3

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 64, 64, 3])

        graph = inference(BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(graph['train_logit'])

        x = tf.placeholder(tf.float32, [None, 64 * 64 * 3])

        # you need to change the directories to yours.
        logs_train_dir = 'D:/train_data/image_data/input_data/ckpt'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)

            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={graph['input_images']: image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                print('This is a triangle with possibility %.6f' % prediction[:, 0])
            elif max_index == 1:
                print('This is a circle with possibility %.6f' % prediction[:, 1])
            elif max_index == 2:
                print('This is a rectangle with possibility %.6f' % prediction[:, 2])
            return max_index


# ------------------------------------------------------------------------
"""
if __name__ == '__main__':
    train_dir = 'D:/train_data/test_data/input_data'
    test_data, test_data_label, val, val_label = get_files(train_dir, 0.01)
    right_num = 0
    false_num = 0
    for i in range(len(test_data)):
        print(test_data[i])
        img = get_image(test_data[i])
        if evaluate_one_image(img) == test_data_label[i]:
            right_num += 1
        else:
            false_num += 1
            if evaluate_one_image(img) == 0:
                print("三角")
            elif evaluate_one_image(img) == 1:
                print("圆形")
            else:
                print("方形")
            print(test_data_label[i])
            # plt.show()

    print(right_num, false_num)
    print("正确率：%f", right_num / (right_num + false_num))

    # img = get_one_image(val)  # 通过改变参数train or val，进而验证训练集或测试集
    # evaluate_one_image(img)
    # ===========================================================================</span>

    # for i in range(len(train))
"""
