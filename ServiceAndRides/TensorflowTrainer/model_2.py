"""
This model consists of only Numerical Features.
The values are one-hot encoded and saved directly.
['zip_encode', 'location_encode', 'community_encode', 'agency_encode',
'complaint_encode', 'afternoon', 'evening', 'morning',
'night', 'Fri-Sat-Sun', 'Mon-Tue', 'Wed-Thu','TimeTaken']
"""
import tensorflow as tf
import shutil
import six

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = []
LABEL = ''
DEFAULTS = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0.0]]
TO_REMOVE = ''

def init(hparams):
    global COLUMNS, LABEL, DEFAULTS, TO_REMOVE
    COLUMNS = ['Unnamed: 0', 'zip_encode', 'location_encode', 'community_encode', 'agency_encode',
                'complaint_encode', 'afternoon', 'evening', 'morning',
                'night', 'Fri-Sat-Sun', 'Mon-Tue', 'Wed-Thu','TimeTaken']

    LABEL = 'TimeTaken'
    TO_REMOVE = 'Unnamed: 0'

    DEFAULTS = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0.0]]

def read_dataset(filename, mode, batch_size):
    def _input_fn():
        
        def _decode_csv(line):
            """Takes the string input tensor and returns a dict of rank-2 tensors."""
            columns = tf.decode_csv(line, record_defaults=DEFAULTS)
            features = dict(zip(COLUMNS, columns))
            #print(features)

            # Remove unused columns.
            features.pop(TO_REMOVE)

            for key, _ in six.iteritems(features):
                if(key != 'TimeTaken'):
                    features[key] = tf.expand_dims(tf.cast(features[key], tf.int32), -1)
                else:
                    features[key] = tf.expand_dims(tf.cast(features[key], tf.float64), -1)
            return features

        # create file path
        #file_path = 'gs://nyc_servicerequest/processedInput/train*'
        file_path = filename
        

        # Create list of files that match pattern (we are currently not using a pattern
        #   such as 1-of-15)
        file_list = tf.gfile.Glob(file_path)

        # Create dataset from file list
        dataset = tf.data.TextLineDataset(file_list).map(_decode_csv)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size=batch_size * 10)
        else:
            num_epochs = 1
            
        iterator = dataset.repeat(num_epochs).batch(batch_size).make_one_shot_iterator()
        features = iterator.get_next()
        return features, features.pop(LABEL)
    
    return _input_fn

    
def serving_input_fn():
    feature_placeholders = {
        'zip_encode': tf.placeholder(tf.int32, None),
        'location_encode': tf.placeholder(tf.int32, None),
        'community_encode': tf.placeholder(tf.int32, None),
        'agency_encode': tf.placeholder(tf.int32, None),
        'complaint_encode': tf.placeholder(tf.int32, None),
        'afternoon': tf.placeholder(tf.int32, None),
        'evening': tf.placeholder(tf.int32, None),
        'morning': tf.placeholder(tf.int32, None),
        'night': tf.placeholder(tf.int32, None),
        'Fri-Sat-Sun': tf.placeholder(tf.int32, None),
        'Mon-Tue': tf.placeholder(tf.int32, None),
        'Wed-Thu': tf.placeholder(tf.int32, None)
    }
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


def train_and_evaluate(output_dir, hparams):
    
    feature_columns = []
    for each in COLUMNS[1:-1]:
        feature_columns.append(
            tf.feature_column.numeric_column(
                key = each,
                dtype=tf.int32
            )
        )

    # used to wrap the model_fn and returns ops necessary to perform training, evaluation, or predictions
    estimator = tf.estimator.LinearRegressor(
                feature_columns = feature_columns,
                model_dir = output_dir
                )

    train_spec = tf.estimator.TrainSpec(
                    input_fn = read_dataset(
                    filename = hparams['train_data_paths'],
                    mode = tf.estimator.ModeKeys.TRAIN,
                    batch_size = hparams['train_batch_size']
                    ),
                    max_steps = hparams['train_steps']
                )
    
    exported = tf.estimator.LatestExporter('exporter', serving_input_fn, exports_to_keep=None)

    #eval_spec consists of computing metrics to judge the performance of the trained model.
    eval_spec = tf.estimator.EvalSpec(
                input_fn = read_dataset(
                    filename = hparams['eval_data_paths'],
                    mode = tf.estimator.ModeKeys.EVAL,
                    batch_size = 1000
                ),
                #start_delay_secs = 60,
                #throttle_secs = 120,
                exporters = exported
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
