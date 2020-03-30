from utils import *
import tensorflow as tf
import numpy as np
from loadData import DataSet
import random

class abpNet(object):

    def __init__(self, category='Mnist', prior=1, vis_step=10, Train_Epochs=200, batch_size=128, z_size=100,
                 langevin_num=20, lr=0.001, theta=1, delta=0.001, history_dir='./', checkpoint_dir='./', logs_dir='./',
                 recon_dir='./', gen_dir='./'):
        self.test = False
        self.prior = prior
        self.category = category
        self.epoch = Train_Epochs
        self.img_size = 28 if (category == 'Fashion-Mnist' or category == 'Mnist') else 64
        self.batch_size = batch_size
        self.z_size = z_size
        self.langevin_num = langevin_num
        self.vis_step = vis_step

        self.lr = lr
        self.theta = theta
        self.delta = delta
        self.channel = 1 if (category == 'Fashion-Mnist' or category == 'Mnist') else 3

        self.history_dir = history_dir
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir
        self.recon_dir = recon_dir
        self.gen_dir = gen_dir

        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_size], name='latent')

        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_size, self.img_size, self.channel],
                                name='image')

    def build_Model(self):
        self.gen = self.Generator(self.z, reuse=False)
        self.langevin, grad = self.langevin(self.z, self.x)

        """
        Loss and Optimizer
        """
        self.gen_loss = self.l2loss(self.gen, self.x)
        self.var = [var for var in tf.trainable_variables() if var.name.startswith('Gen')]

        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.grad = self.optimizer.compute_gradients(self.gen_loss, var_list=self.var)
        self.compute_grad = self.optimizer.apply_gradients(self.grad)
        """
        Logs
        """
        tf.summary.scalar('gen_loss', tf.reduce_mean(self.gen_loss))
        # TODO showing specifically
        # tf.summary.histogram('hyper params', self.hyper_var)
        self.summary_op = tf.summary.merge_all()


    def langevin(self, z_arg, x):
        def cond(i, z, grad):
            return tf.less(i, self.langevin_num)

        def body(i, z, grad):
            noise = tf.random_normal(shape=[self.batch_size, self.z_size], name='noise')
            gen = self.Generator(z, reuse=True)
            gen_loss = self.l2loss(gen, x)
            grad = tf.gradients(gen_loss, z, name='gen_grad')[0]

            z = z - 0.5 * self.delta * self.delta * (grad + self.prior*z) + self.delta*noise
            # x = x - 0.5 * self.config.delta2 * self.config.delta2 * (prior + grad) + 0.001*noise

            return tf.add(i, 1), z, grad

        with tf.name_scope("langevin_dynamic"):
            i = tf.constant(0)
            grad = tf.constant(0, shape=(list(z_arg.shape)), dtype=tf.float32)
            i, z, grad = tf.while_loop(cond, body, [i, z_arg, grad])

            return z, tf.reduce_mean(grad)

    def Generator(self, z, reuse=False):
        with tf.variable_scope('Gen', reuse=reuse):
            if self.category == 'Fashion-Mnist' or self.category == 'Mnist':
                z = tf.reshape(z, [-1, self.z_size])

                fc1 = tf.layers.dense(inputs=z, units=1024, name='fc1')

                fc1 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(fc1, is_training=True))

                fc2 = tf.layers.dense(inputs=fc1, units=1568, name='fc2')

                fc2 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(fc2, is_training=True))

                fc2 = tf.reshape(fc2, [self.batch_size, 7, 7, -1])

                dc1 = deconv2d(fc2, (self.batch_size, self.img_size // 2, self.img_size // 2, 128), kernal=(5, 5),
                               name='dc1')
                dc1 = tf.contrib.layers.batch_norm(dc1, is_training=True)
                dc1 = tf.nn.leaky_relu(dc1)

                dc2 = deconv2d(dc1, (self.batch_size, self.img_size // 1, self.img_size // 1, 1), kernal=(5, 5),
                               name='dc2')

                output = tf.nn.tanh(dc2)

            return output

    def l2loss(self, syn, obs):
        a = 1.0 / (2 * self.theta * self.theta) * tf.square(syn - obs)
        # a = tf.reduce_sum(a, axis=-1)
        # a = tf.reduce_sum(a, axis=-1)
        l2loss = tf.reduce_mean(tf.reduce_sum(a,axis=(1,2,3)), axis=0)
        # l2loss = tf.reduce_mean(1.0 / (2 * self.theta * self.theta) * tf.square(syn - obs), axis=0)
        return l2loss

    def train(self, sess):
        self.build_Model()

        data = DataSet(img_size=self.img_size, batch_size=self.batch_size, category=self.category)

        sess.run(tf.global_variables_initializer())


        saver = tf.train.Saver(max_to_keep=10)

        writer = tf.summary.FileWriter(self.logs_dir, sess.graph)

        start = 0
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)

        if latest_checkpoint:
            latest_checkpoint.split('-')
            start = int(latest_checkpoint.split('-')[-1])
            saver.restore(sess, latest_checkpoint)
            print('Loading checkpoint {}.'.format(latest_checkpoint))

        tf.get_default_graph().finalize()

        latent_gen = np.random.normal(size=(len(data), self.z_size))

        for epoch in range(start + 1, self.epoch):
            num_batch = int(len(data) / self.batch_size)
            losses = []
            for step in range(num_batch):
                obs = data.NextBatch(step)
                z = latent_gen[step * self.batch_size: (step + 1) * self.batch_size].copy()
                z = sess.run(self.langevin, feed_dict={self.z: z, self.x: obs})

                l2loss, summary, _ = sess.run([self.gen_loss, self.summary_op, self.compute_grad],
                                              feed_dict={self.z: z, self.x: obs})
                latent_gen[step * self.batch_size: (step + 1) * self.batch_size] = z
                losses.append(l2loss)
                writer.add_summary(summary, global_step=epoch)

            print(epoch, ": Loss : ", np.mean(losses))
            if epoch % self.vis_step == 0:
                self.visualize(saver, sess, len(data), epoch, latent_gen, data)

    def visualize(self, saver, sess, num_data, epoch, latent_gen, data):
        saver.save(sess, "%s/%s" % (self.checkpoint_dir, 'model.ckpt'), global_step=epoch)
        idx = random.randint(0, int(num_data / self.batch_size) - 1)
        z = latent_gen[idx * self.batch_size: (idx + 1) * self.batch_size]
        """
                Recon
        """
        obs = data.NextBatch(idx)
        z = sess.run(self.langevin, feed_dict={self.z: z, self.x: obs})
        sys = sess.run(self.gen, feed_dict={self.z: z})
        sys = np.array((sys + 1) * 127.5, dtype=np.float)
        path = self.recon_dir + 'epoch' + str(epoch) + 'recon.jpg'
        # show_z_and_img(epoch, path, z, sys, self.row, self.col)
        show_in_one(path, sys, column=16, row=8)

        """
        Generation
        """
        # obs = data.NextBatch(idx, test=True)
        z = np.random.normal(size=(self.batch_size, self.z_size))
        # z = sess.run(self.langevin, feed_dict={self.z: z, self.x: obs})
        sys = sess.run(self.gen, feed_dict={self.z: z})
        sys = np.array((sys + 1) * 127.5, dtype=np.float)
        path = self.gen_dir + 'epoch' + str(epoch) + 'gens.jpg'
        show_in_one(path, sys, column=16, row=8)