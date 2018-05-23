# 导入文件
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util

from build_train_model import inference, losses, trainning, evaluation, X, Y
from process_input_pic import get_batch, get_files, list_to_iterator, get_nparray_batch
from val_train_model.val_train_model import get_one_image, get_image, evaluate_one_image
from val_train_model.val_train_model_from_pb import recognize

# 变量声明
N_CLASSES = 3
IMG_W = 64  # resize图像，太大的话训练时间久
IMG_H = 64
BATCH_SIZE = 20
CAPACITY = 200
MAX_STEP = 300  # 一般大于10K
learning_rate = 0.0005  # 一般小于0.0001

# 获取批次batch
train_dir = 'D:/train_data/image_data/input_data'  # 训练样本的读入路径
logs_train_ckpt_dir = 'D:/train_data/image_data/input_data/ckpt'  # logs存储路径
logs_train_pb_dir = 'D:/train_data/image_data/input_data/pb/output.pb'

#提取文件
train, train_label, val, val_label = get_files(train_dir, 0.2)

# 训练操作定义
train_logits = inference(BATCH_SIZE, N_CLASSES)
train_loss = losses(train_logits, batch_size=BATCH_SIZE)
train_op = trainning(train_loss, learning_rate)
train_acc = evaluation(train_logits, batch_size=BATCH_SIZE)


# 产生一个会话
sess = tf.Session()
# 产生一个writer来写log文件
train_writer = tf.summary.FileWriter(logs_train_ckpt_dir, sess.graph)
# val_writer = tf.summary.FileWriter(logs_test_dir, sess.graph)
# 产生一个saver来存储训练好的模型
saver = tf.train.Saver()
# 所有节点初始化
sess.run(tf.global_variables_initializer())


# 进行batch的训练
try:
    # 执行MAX_STEP步的训练，一步一个batch
    for step in np.arange(MAX_STEP):
        try:
            image_batch, label_batch = get_nparray_batch(
                train_iter, train_label_iter, IMG_W, IMG_H, 3, BATCH_SIZE)
        except:
            train_iter, train_label_iter = list_to_iterator(train, train_label)
            image_batch, label_batch = get_nparray_batch(
                train_iter, train_label_iter, IMG_W, IMG_H, 3, BATCH_SIZE)
            print("获取完了")

        _ = sess.run(train_op, feed_dict={X: image_batch, Y: label_batch})
        tra_loss = sess.run(train_loss, feed_dict={
            X: image_batch, Y: label_batch})
        tra_acc = sess.run(train_acc, feed_dict={
                           X: image_batch, Y: label_batch})

        # 每隔50步打印一次当前的loss以及acc
        if step % 10 == 0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %
                  (step, tra_loss, tra_acc * 100.0))

        # 每隔100步，保存一次训练好的模型
        if (step + 1) == MAX_STEP:
            #保存到ckpt文件
            checkpoint_path = os.path.join(logs_train_ckpt_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)


            # 保存到pb文件
            # 得到当前的图的 GraphDef 部分，通过这个部分就可以完成重输入层到输出层的计算过程
            graph_def = tf.get_default_graph().as_graph_def()

            output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
                sess,
                graph_def,
                ["softmax_linear/softmax"]  # 需要保存节点的名字
            )

            with tf.gfile.GFile(logs_train_pb_dir, "wb") as f:  # 保存模型
                f.write(output_graph_def.SerializeToString())  # 序列化输出
            print("%d ops in the final graph." % len(output_graph_def.node))


except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')



# 验证结果


right_num = 0
false_num = 0
for i in range(len(val)):
    print(val[i])

    if recognize(val[i],
                 "D:/train_data/image_data/input_data/pb/output.pb") == val_label[i]:
        right_num += 1
    else:
        false_num += 1

print(right_num, false_num)
print("正确率：%f", right_num / (right_num + false_num))
