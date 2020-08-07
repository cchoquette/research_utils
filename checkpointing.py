import tensorflow as tf
from tensorflow.keras import callbacks
import command_line

parser = command_line.get_parser()


class CheckpointCallback(callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    self.model.epoch.assign_add(1)
    self.model.manager.save()


class CheckpointingModel(tf.keras.models.Model):
  def __init__(self, checkpoint_dir=None, max_to_keep=10, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.checkpoint_dir = checkpoint_dir
    self.max_to_keep = max_to_keep
    self.epoch = None
    self.manager = None
    self.ckpt = None

  def compile(self, optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              **kwargs):
    super().compile(optimizer=optimizer,
                    loss=loss,
                    metrics=metrics,
                    loss_weights=loss_weights,
                    weighted_metrics=weighted_metrics,
                    **kwargs)
    self.epoch = tf.Variable(0)
    self.ckpt = tf.train.Checkpoint(epoch=self.epoch, optimizer=optimizer, net=self)
    self.manager = tf.train.CheckpointManager(self.ckpt, f'/checkpoint/cchoquet/{self.checkpoint_dir}',
                                         max_to_keep=self.max_to_keep)
    if self.manager.latest_checkpoint:
      self.ckpt.restore(self.manager.latest_checkpoint)
      print(f"restoring from: {self.manager.latest_checkpoint}")
    else:
      print(f"No checkpoint found in: {self.checkpoint_dir}")

  def fit(self,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_batch_size=None,
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False):
    if callbacks is None:
      callbacks = []
    callbacks.append(CheckpointCallback())
    if self.epoch < epochs:
      out = super().fit(x=x,
                  y=y,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=verbose,
                  callbacks=callbacks,
                  validation_split=validation_split,
                  validation_data=validation_data,
                  shuffle=shuffle,
                  class_weight=class_weight,
                  sample_weight=sample_weight)
    else:
      out = None
    return out
