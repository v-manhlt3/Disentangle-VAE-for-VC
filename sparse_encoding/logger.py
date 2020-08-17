import tensorflow as tf
import numpy as np
import scipy.misc
from tensorboardX import SummaryWriter

class Logger(object):
    
    def __init__(self, log_dir ):
        """Create a summary writer logging to log_dir."""
        
        # self.train_writer = tf.summary.FileWriter(log_dir + "/train")
        # self.test_writer = tf.summary.FileWriter(log_dir + "/eval")
        self.writer = SummaryWriter(log_dir)

        # self.loss = tf.Variable(0.0)
        # tf.summary.scalar("loss", self.loss)

        # self.merged = tf.summary.merge_all()

        # self.session = tf.InteractiveSession()
        # self.session.run(tf.global_variables_initializer())


    # def scalar_summary(self, train_loss, test_loss, step):
    #     """Log a scalar variable."""

    #     summary = self.session.run(self.merged, {self.loss: train_loss})
    #     self.train_writer.add_summary(summary, step) 
    #     self.train_writer.flush()

    #     summary = self.session.run(self.merged, {self.loss: test_loss})
    #     self.test_writer.add_summary(summary, step) 
    #     self.test_writer.flush()

    def write_log(self, recons_loss, prior_loss, epoch):
        print('====> Reconstruction Loss: ', recons_loss)
        print('====> Prior Loss: ', prior_loss)
        self.writer.add_scalar('Loss\Reconstruction Loss', recons_loss, epoch)
        self.writer.add_scalar('Loss\Prior Loss', prior_loss, epoch)
        
