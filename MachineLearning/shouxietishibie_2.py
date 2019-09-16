import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)

sess = tf.InteractiveSession()

#构建Softmax回归模型

x = tf.placeholder("float",shape=[None,784])
y_ = tf.placeholder("float",shape=[None,10])

#在机器学习的应用过程中，模型参数一般用Varibale来表示
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#变量需要通过session初始化后才能在 session中使用。这一步骤为，为初始值指定具体值（本例当中是全为0），并将其分配给每个变量，可以一次性为所有变量完成操作

sess.run(tf.initialize_all_variables())

#类别预测与损失函数
#Softmax函数-回归模型

y = tf.nn.softmax(tf.matmul(x,W) + b)

#交叉熵--损失函数
#这里的tf.reduce_sum是将每个图片的交叉熵值都加起来了。
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#使用梯度下降，快速下降法让交叉熵下降，步长为0.01

# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#每一步迭代都会加载50个训练样本，然后每执行一次train_step，并通过feed_dict将x和y_张量占位符用训练数据代替
# for i in range(1000):
#     batch = mnist.train.next_batch(50)
#     train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#评估模型

# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


#构建一个多卷积神经网络
#使用的神经元是 ReLU神经元（线性整流函数） 即 f（x） = max(0,x)  ,在此处应该是 y = max(0,w*x+b)

#权重初始化
#由于使用的是ReLU神经元，因此较好的做法是用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题。

#为了不在建立模型的时候反复做初始化操作，我们定义两个函数用于初始化，分别用于初始化 权重W和偏置b

#函数 tf.truncated_normal(shape,mean,stddev,detype,seed,name); 返回值是 从一个截断的正态分布里输出随机数，这些数满足指定的平均值和标准差的正态分布
#shape:生成的张量的维度；mean：截断正态分布的平均值；stddev：正太分布的标准差；detype：生成数据的类型；seed：为正态分布创建一个随机的种子；name：操作的名称
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
#tf.constant(value,detype,shape,name,verify_shape).返回值 根据value的值生成一个shape维度的常量张量 、
# value：输出张量的值；shape：输出张量的维度
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#卷积和池化
#在下面的实例中，直接使用原版。我们的卷积使用1步长（stride size），0边距（padding size）的模板，以保证输出和输入是同一个大小。
# 池化用最简单传统的2×2大小的模板做max pooling --主要操作是保留特征的完整性进行降维操作以及减少参数：如进行 旋转、平移、缩小


# tf.nn.conv2d(input,filter,strides,padding,use_cudnn_on_gpu=Nne)
# 1、input：输入的要做卷积的图片，要求为1个张量shape [batch, in_height, in_weight, in_channel]，其中batch为图片的数量，in_height为图片的高度
# in_weight 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3。
# 2、filter： 卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，
# 其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，
# out_channel 是卷积核数量。
# 3、strides： 卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
# 4、padding： string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。
# "SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
# 5、use_cudnn_on_gpu： bool类型，是否使用cudnn加速，默认为true

#卷积
# tf.nn.conv2d(input,filter,strides,padding,use_cudnn_on_gpu=Nne)
# 1、input：输入的要做卷积的图片，要求为1个张量shape [batch, in_height, in_weight, in_channel]，其中batch为图片的数量，in_height为图片的高度
# in_weight 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3。
# 2、filter： 卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，
# 其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，
# out_channel 是卷积核数量。
# 3、strides： 卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
# 4、padding： string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。
# "SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
# 5、use_cudnn_on_gpu： bool类型，是否使用cudnn加速，默认为true
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#池化
#tf.nn.max_pool(value, ksize, strides, padding, name=None)
#第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
#第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
#第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
#第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'
#返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#第一层卷积

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x,[-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#密集连接层

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout--降低过拟合

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#训练和评估模型

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print ("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


