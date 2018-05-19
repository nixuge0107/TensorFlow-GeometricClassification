1.  triangle
2.  circle
3.  rectangle
# 结果
357 22

正确率：%f 0.941952506596306

# 思路
## batch操作
### 需求
- 传递给`sess.run`的图片格式为[None,64&times;64&times;3]
- 传递给输入`inference()`函数的`image`格式为`x = tf.reshape(X, shape=[-1, 64, 64, 3])`或者batch
