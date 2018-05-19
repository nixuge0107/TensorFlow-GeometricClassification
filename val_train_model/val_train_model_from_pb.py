import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def recognize(jpg_path, pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        img = Image.open(jpg_path)
        # plt.imshow(img)
        # plt.show()
        image_array = np.array(img.resize([64, 64]))
        image_array = image_array.flatten() / 255.0
        batch_x = np.zeros([1, 64 * 64 * 3])
        batch_x[0, :] = image_array

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            input_x = sess.graph.get_tensor_by_name("input:0")
            print(input_x)
            out_softmax = sess.graph.get_tensor_by_name(
                "softmax_linear/softmax:0")
            print(out_softmax)
            # out_label = sess.graph.get_tensor_by_name("softmax_linear/output:0")
            # print(out_label)

            img_out_softmax = sess.run(
                out_softmax, feed_dict={input_x: batch_x})

            print("img_out_softmax:", img_out_softmax)
            prediction_labels = np.argmax(img_out_softmax, axis=1)
            print("label:", prediction_labels)
            max_index = prediction_labels
            if max_index == 0:
                print('This is a triangle with possibility %.6f' %
                      img_out_softmax[:, 0])
            elif max_index == 1:
                print('This is a circle with possibility %.6f' %
                      img_out_softmax[:, 1])
            elif max_index == 2:
                print('This is a rectangle with possibility %.6f' %
                      img_out_softmax[:, 2])
            return max_index


if __name__ == '__main__':
    recognize("D:/train_data/image_data/input_data/triangle/525samples1.jpg",
              "D:/train_data/image_data/input_data/pb/output.pb")
