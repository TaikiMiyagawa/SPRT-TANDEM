import os
import tensorflow as tf

class TensorboardLogger():
    """ 
    Usage:
        # Example code (TF2.0.0)
        global_step = np.array(0, dtype=np.int64)
        tf.summary.experimental.set_step(global_step)
        tblogger = TensorboardLogger(logdir)
        with tblogger.writer.as_default():
            tblogger.scalar_summary(tab, value, description)
    """
    def __init__(self, root_tblogs, subproject_name, exp_phase, comment, time_stamp):
        """Create a summary writer logging to root_tblogs + naming rule shown below.
        Args:
            root_tblogs: A string. 
            subproject_name: A string.
            comment: A string.
            time_stamp: A str of a time stamp. e.g., time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        Remark:
            Tensorboard logs of one run will be saved in "root_tblogs/subproject_name_exp_phase/comment_time_stamp"            """

        # Naming Rule
        self.root_tblogs = root_tblogs
        self.subproject_name = subproject_name
        self.exp_phase = exp_phase
        self.comment = comment
        self.time_stamp = time_stamp
        self.dir_tblogs = self.root_tblogs + "/" + self.subproject_name + "_" + self.exp_phase + "/" + self.comment + "_"+ self.time_stamp
        if not os.path.exists(self.dir_tblogs):
            os.makedirs(self.dir_tblogs)
        print("Set Tensorboard drectory: ", self.dir_tblogs)

        # Create a summary writer
        self.writer = tf.summary.create_file_writer(self.dir_tblogs, flush_millis=10000)

    def scalar_summary(self, tag, value, step=None, description=None):
        """Log a scalar variable.
           Invoke in writer.as_default() context."""
        tf.summary.scalar(name=tag, data=value, step=step, description=description)
