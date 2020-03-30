# _author_ = "jiali cui"
# Email : cuijiali961224@gmail.com
# Data:
from abp_net import abpNet
import tensorflow as tf
import os
import shutil
from save import saveModule

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_integer('prior', 1, 'prior for langevin')


tf.flags.DEFINE_integer('vis_step', 100, 'Epoch when visualize')
tf.flags.DEFINE_integer('Train_Epochs', 1000, 'Number of Epochs to train')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch size of training images')
tf.flags.DEFINE_integer('z_size', 200, 'dimensions of latent factors')
tf.flags.DEFINE_integer('langevin_num', 30, 'Sample steps for Langevin dynamics')
tf.flags.DEFINE_float('lr', 0.0001, 'learning rate')
tf.flags.DEFINE_float('theta', 0.1, 'learning rate')
tf.flags.DEFINE_float('delta', 0.1, 'learning rate')

tf.flags.DEFINE_string('category', 'Mnist', 'DataSet category')

tf.flags.DEFINE_string('history_dir', './output/history/', 'history')
tf.flags.DEFINE_string('checkpoint_dir', './output/checkpoint/', 'checkpoint')
tf.flags.DEFINE_string('logs_dir', './output/logs/', 'logs')
tf.flags.DEFINE_string('recon_dir', './output/recon/', 'recon')
tf.flags.DEFINE_string('gen_dir', './output/gens/', 'gen')


def main(_):
    saveFlag = True
    saveFlag = False
    if saveFlag:
        save = saveModule(category=FLAGS.category,
                          prior=FLAGS.prior,
                          z_size=FLAGS.z_size,
                          langevin_num=FLAGS.langevin_num,
                          lr=FLAGS.lr,
                          theta=FLAGS.theta,
                          delta=FLAGS.delta)
        save.process()

    else:
        model = abpNet(
            prior=FLAGS.prior,
            category=FLAGS.category,
            vis_step=FLAGS.vis_step,
            Train_Epochs=FLAGS.Train_Epochs,
            batch_size=FLAGS.batch_size,
            z_size=FLAGS.z_size,
            langevin_num=FLAGS.langevin_num,
            lr=FLAGS.lr,
            theta=FLAGS.theta,
            delta=FLAGS.delta,
            history_dir=FLAGS.history_dir,
            checkpoint_dir=FLAGS.checkpoint_dir,
            logs_dir=FLAGS.logs_dir,
            recon_dir=FLAGS.recon_dir,
            gen_dir=FLAGS.gen_dir,
        )

        continueTrain = False
        # continueTrain = True
        with tf.Session() as sess:
            if not continueTrain:
                if os.path.exists(FLAGS.checkpoint_dir):
                    shutil.rmtree(FLAGS.checkpoint_dir[:-1])
                os.makedirs(FLAGS.checkpoint_dir)

            if os.path.exists(FLAGS.logs_dir):
                shutil.rmtree(FLAGS.logs_dir[:-1])
            os.makedirs(FLAGS.logs_dir)

            if not os.path.exists(FLAGS.history_dir):
                os.makedirs(FLAGS.history_dir)

            if os.path.exists(FLAGS.recon_dir):
                shutil.rmtree(FLAGS.recon_dir[:-1])
            os.makedirs(FLAGS.recon_dir)

            if os.path.exists(FLAGS.gen_dir):
                shutil.rmtree(FLAGS.gen_dir[:-1])
            os.makedirs(FLAGS.gen_dir)

            model.train(sess)


if __name__ == '__main__':
    tf.app.run()
