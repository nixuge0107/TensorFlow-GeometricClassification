# 导入文件
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util

from build_train_model import inference, losses, trainning, evaluation, X, Y
from process_input_pic import get_batch, get_files, list_to_iterator, get_nparray_batch
from val_train_model.val_train_model import get_one_image, get_image, evaluate_one_image

# 变量声明
N_CLASSES = 3
IMG_W = 64  # resize图像，太大的话训练时间久
IMG_H = 64
BATCH_SIZE = 40
CAPACITY = 200
MAX_STEP = 100  # 一般大于10K
learning_rate = 0.11010  # 一般小于0.0001

# 获取批次batch
train_dir = 'D:/train_data/image_data/input_data'  # 训练样本的读入路径
logs_train_ckpt_dir = 'D:/train_data/image_data/input_data/ckpt'  # logs存储路径
logs_train_pb_dir = 'D:/train_data/image_data/input_data/pb/output.pb'
# logs_test_dir =  'E:/Re_train/image_data/test'        #logs存储路径

# train, train_label = input_data.get_files(train_dir)
train, train_label, val, val_label = get_files(train_dir, 0.2)
# 将list转化为iter
# train_iter, train_label_iter = list_to_iterator(train, train_label)
# 训练数据及标签
train_batch, train_label_batch = get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
# 测试数据及标签
# val_batch, val_label_batch = get_batch(val, val_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)


# 训练操作定义
train_logits = inference(BATCH_SIZE, N_CLASSES)
train_loss = losses(train_logits, batch_size=BATCH_SIZE)
train_op = trainning(train_loss, learning_rate)
train_acc = evaluation(train_logits, batch_size=BATCH_SIZE)
# # 测试操作定义
# test_logits = inference(val_batch, BATCH_SIZE, N_CLASSES)
# test_loss = losses(test_logits, val_label_batch)
# test_acc = evaluation(test_logits, val_label_batch)


# 这个是log汇总记录
summary_op = tf.summary.merge_all()

# 产生一个会话
sess = tf.Session()
# 产生一个writer来写log文件
train_writer = tf.summary.FileWriter(logs_train_ckpt_dir, sess.graph)
# val_writer = tf.summary.FileWriter(logs_test_dir, sess.graph)
# 产生一个saver来存储训练好的模型
saver = tf.train.Saver()
# 所有节点初始化
sess.run(tf.global_variables_initializer())
# 队列监控
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 进行batch的训练
try:
    # 执行MAX_STEP步的训练，一步一个batch
    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break
        # 获取batch数据
        try:
            image_batch, label_batch = get_nparray_batch(train_iter, train_label_iter, IMG_W, IMG_H, 3, BATCH_SIZE)
        except:
            train_iter, train_label_iter = list_to_iterator(train, train_label)
            image_batch, label_batch = get_nparray_batch(train_iter, train_label_iter, IMG_W, IMG_H, 3, BATCH_SIZE)
            print("获取完了")

        # 启动以下操作节点，有个疑问，为什么train_logits在这里没有开启？
        # _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc],
        #                                 feed_dict={GRAPH['input_images']: train_batch})
        _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc], feed_dict={X: image_batch, Y: label_batch})

        # 每隔50步打印一次当前的loss以及acc，同时记录log，写入writer
        if step % 10 == 0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
            # summary_str = sess.run(summary_op)
            # train_writer.add_summary(summary_str, step)
        # 每隔100步，保存一次训练好的模型
        if (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_ckpt_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
            # 保存到pb文件
            graph_def = tf.get_default_graph().as_graph_def()  # 得到当前的图的 GraphDef 部分，通过这个部分就可以完成重输入层到输出层的计算过程
            # 打开一个文件
            fo = open("foo.txt", "w")
            fo.write(str(graph_def))
            # 关闭打开的文件
            fo.close()

            output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
                sess,
                graph_def,
                ["softmax_linear/softmax"]  # 需要保存节点的名字
            )
            f1 = open("f11.txt", "w")
            f1.write(str(output_graph_def))
            # 关闭打开的文件
            f1.close()
            with tf.gfile.GFile(logs_train_pb_dir, "wb") as f:  # 保存模型
                f.write(output_graph_def.SerializeToString())  # 序列化输出
            print("%d ops in the final graph." % len(output_graph_def.node))


except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')

finally:
    coord.request_stop()

"""
验证结果
"""

# right_num = 0
# false_num = 0
# for i in range(len(val)):
#     print(val[i])
#     img = get_image(val[i])
#     img = img.flatten() / 255.0
#     if evaluate_one_image(img) == val_label[i]:
#         right_num += 1
#     else:
#         false_num += 1
#         if evaluate_one_image(img) == 0:
#             print("三角")
#         elif evaluate_one_image(img) == 1:
#             print("圆形")
#         else:
#             print("方形")
#         print(val_label[i])
#         # plt.show()
#
# print(right_num, false_num)
# print("正确率：%f", right_num / (right_num + false_num))
