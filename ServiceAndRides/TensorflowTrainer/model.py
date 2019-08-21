"""
This model contains both Categorical and Numerical Features.
Preprocessing is done by Tensorflow functions itself and then values are passed onto the model.
"""
import tensorflow as tf
import shutil
import six

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = []
LABEL = ''
DEFAULTS = [[0], [''], [''], [''], [''], [''], [''], [''], [0.0]]
TO_REMOVE = ''

def init(hparams):
    global COLUMNS, LABEL, DEFAULTS, TO_REMOVE
    COLUMNS = ['Unnamed: 0', 'day_period', 'day_of_week', 'zip_encode', 'location_encode',
                'community_encode', 'agency_encode', 'complaint_encode', 'TimeTaken']

    LABEL = 'TimeTaken'
    TO_REMOVE = 'Unnamed: 0'

    DEFAULTS = [[0], [''], [''], [''], [''], [''], [''], [''], [0.0]]

def read_dataset(filename, mode, batch_size):
    def _input_fn():
        
        def _decode_csv(line):
            """Takes the string input tensor and returns a dict of rank-2 tensors."""
            columns = tf.decode_csv(line, record_defaults=DEFAULTS)
            features = dict(zip(COLUMNS, columns))

            # Remove unused columns.
            features.pop(TO_REMOVE)

            for key, _ in six.iteritems(features):
                if(key != 'TimeTaken'):
                    features[key] = tf.expand_dims(features[key], -1)
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

        """dataset = (tf.data.TextLineDataset(file_list)  # Read text file
                   .skip(1) 
                   .map(decode_csv))  # Transform each elem by applying decode_csv fn
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size = 10 * batch_size)
        else:
            num_epochs = 1 # end-of-input after this
            
        iterator = dataset.repeat(num_epochs).batch(batch_size).make_one_shot_iterator()
        features = iterator.get_next()
        return features, features.pop(LABEL)
        """
    return _input_fn

    
def serving_input_fn():
    feature_placeholders = {
        'day_period': tf.placeholder(tf.string, [None]),
        'day_of_week': tf.placeholder(tf.string, [None]),
        'zip_encode': tf.placeholder(tf.string, [None]),
        'location_encode': tf.placeholder(tf.string, [None]),
        'community_encode': tf.placeholder(tf.string, [None]),
        'agency_encode': tf.placeholder(tf.string, [None]),
        'complaint_encode': tf.placeholder(tf.string, [None]) 
    }
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

def train_and_evaluate(output_dir, hparams):
    unique_vals = dict()
    unique_vals['day_period'] = ['morning', 'afternoon', 'evening', 'night']
    unique_vals['day_of_week'] = ['Mon-Tue', 'Wed-Thu', 'Fri-Sat-Sun']
    unique_vals['zip_encode'] = ['zip_bin1', 'zip_bin2', 'zip_bin3', 'zip_bin4']
    unique_vals['location_encode'] = ['location_bin1', 'location_bin2', 'location_bin3', 'location_bin4']
    unique_vals['community_encode'] = ['community_bin1', 'community_bin2', 'community_bin3']
    unique_vals['agency_encode'] =  ['agency_bin1', 'agency_bin2', 'agency_bin3', 'agency_bin4', 'agency_bin5', 'agency_bin6']
    unique_vals['complaint_encode'] = ['complaint_bin1', 'complaint_bin2', 'complaint_bin3']

    feature_columns = []
    for each in COLUMNS[1:-1]:
        feature_columns.append(
            tf.feature_column.categorical_column_with_vocabulary_list(
                key = each,
                vocabulary_list = unique_vals[each]
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
