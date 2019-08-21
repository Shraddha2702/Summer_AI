import traceback
import argparse
import json
import os
import tensorflow as tf

import model ###Change this if your model file changes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
  # Input Arguments
    parser.add_argument(
      '--train_data_paths',
      help='GCS or local path to training data',
      required=True
  )
    parser.add_argument(
      '--eval_data_paths',
      help='GCS or local path to evaluation data',
      required=True
  )
    parser.add_argument(
      '--train_batch_size',
      help='Batch size for training steps',
      type=int,
      default=20000
  )
    parser.add_argument(
      '--train_steps',
      help='Steps to run the training job for. A step is one batch-size',
      type=int,
      default=1
  )
    parser.add_argument(
      '--output_dir',
      help='GCS location to write checkpoints and export models',
      required=True
  )
    parser.add_argument(
      '--job-dir',
      help='this model ignores this field, but it is required by gcloud',
      default='junk'
  )

    args = parser.parse_args()
    hparams = args.__dict__
  
  # unused args provided by service
    job_dir = hparams.pop('job_dir')
    hparams.pop('job-dir', None)

    output_dir = hparams.pop('output_dir')

  # This code can be removed if you are not using hyperparameter tuning
    output_dir = os.path.join(
      output_dir,
      json.loads(
          os.environ.get('TF_CONFIG', '{}')
      ).get('task', {}).get('trial', '')
  )

  # calculate train_steps if not provided
    if hparams['train_steps'] < 1:
       # 1,000 steps at batch_size of 100
        hparams['train_steps'] = (1000 * 100) // hparams['train_batch_size']
        print ("Training for {} steps".format(hparams['train_steps']))
  
    model.init(hparams)
  
  # Run the training job
    model.train_and_evaluate(output_dir, hparams)
