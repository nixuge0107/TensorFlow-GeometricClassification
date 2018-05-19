import tensorflow as tf
import os
from tensorflow.python.tools import freeze_graph

model_path = "D:/train_data/image_data/input_data/ckpt/model.ckpt-9"  # 设置model的路径，因新版tensorflow会生成三个文件，只需写到数字前


def main():


    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)

        # 保存图
        tf.train.write_graph(sess.graph_def, 'output_model/pb_model', 'model.pb')
        # 把图和参数结构一起
        freeze_graph.freeze_graph('output_model/pb_model/model.pb', '', False, model_path, 'out', 'save/restore_all',
                                  'save/Const:0', 'output_model/pb_model/frozen_model.pb', False, "")

    print("done")


if __name__ == '__main__':
    freeze_graph.freeze_graph('D:/train_data/image_data/input_data/pb/output.pb', '', True, model_path, 'softmax_linear/softmax:0', 'save/restore_all',
                                  'save/Const:0', 'D:/train_data/image_data/input_data/pb/frozen_model.pb', False, "")
