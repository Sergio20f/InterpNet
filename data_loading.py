import tensorflow as tf
import tensorflow_datasets as tfds
from helpers import normalize_img


class Data:
    """
    Class to import a specific dataset and build training, test, and validation datasets.

    Attributes:
    -----------
    - name (str) : tensorflow_datasets dataset name.
    - batch_size (int) = integer value for the batch size, default is 128.
    norm_function (function): Function that will be mapped across the datasets in order to normalise them.
    - resize (False or tuple): If false there is no resizing. If a tuple is given as an input, the images across the set
    will be resized (only works for datasets with images).
    - custom_dir (False or str): If the user uses a default dataset from tfds then this parameter should be kept as
    False. Otherwise, it should be a string containing the path to the custom dataset. This custom dataset should
    contain folders called "train" and "test" with the corresponding data points.

    Methods:
    -----------
    test_data_prep(): generates independent and different training and validation datasets of normalised images and
    applies cache(), batch(), and prefetch().
    train_data_prep(): generates a training dataset of normalised images and applies cache(), batch(), and prefetch().
    """

    def __init__(self, name: str, batch_size=128, norm_func=normalize_img, resize=False, custom_dir=False): # ignore custom_dir
        """
        Initialises the class and creates the global variables.

        :param name: tensorflow_datasets dataset name.
        :type name: str
        :param batch_size: integer value for the batch size, default is 128.
        :type batch_size: int
        :param resize: Parameter corresponding to the new dimensions of the images contained in the dataset after resizing. If 
                       False then the images in the dataset are not resized.
        :param norm_func: Function that will be mapped across the datasets in order to normalise them.
        :type norm_func: function
        :param resize: If false there is no resizing. If a tuple is given as an input, the images across the set
                       will be resized (only works for datasets with images).
        :param custom_dir: If the user uses a default dataset from tfds then this parameter should be kept as
                           False. Otherwise, it should be a string containing the path to the custom dataset.
                           This custom dataset should contain folders called "train" and "test" with the corresponding
                           data points.
        """
        self.batch_size = batch_size
        self.name = name
        self.norm_func = norm_func
        self.resize = resize
        self.custom_dir = custom_dir

        if custom_dir:
            builder = tfds.ImageFolder(custom_dir)
            self.builder = builder

            # Here for efficiency, otherwise we would be calling builder in each iteration of training_loop
            train_data = self.builder.as_dataset(split="train", shuffle_files=True, as_supervised=True)
            self.train_data = train_data

        pass

    def train_data_prep(self, data_type="image"):
        """
        Method that generates a training dataset of normalised images and applies cache(), batch(), and prefetch(). It
        uses the method "ReadInstruction()" from tfds.core to load the desired amount of data in each training
        iteration. This desired amount of data is determined by the parameter dts.

        :param dts: Parameter that determines the quantity of data going into the output training dataset.
        :type dts: int
        :param data_type: (Will use in the future to determine the datatype of the data points in the target dataset).
        :type data_type: str
        :param norm_func: Takes the value of None for whenever we do not want to apply any normalisation function to our
        data. If not None, then it must be a python function.
        :type norm_func: function

        :return: Training data.
        """
        # if self.custom_dir:
            # ReadInstruction is not yet supported for ImageFolder
            # train_data = self.builder.as_dataset(split=tfds.core.ReadInstruction("train",
            #                                                                     from_=0, to=dts, unit="abs"),
            #                                     shuffle_files=True, as_supervised=True)

            # Make dts play a role for custom datasets
        #    train_data = self.train_data.take(dts)

        # else:
        train_data, ds_info = tfds.load(self.name,
                                        split=["test", "train[0%:10%]", "train[10%:]"], # test - validation - train
                                        shuffle_files=True,
                                        as_supervised=True,
                                        with_info=True)

        if self.norm_func:
            train_data = train_data.map(self.norm_func,
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
            
        if self.resize:
            train_data = train_data.map(
                lambda image, label: (tf.image.resize(image, [self.resize[0], self.resize[1]]), label))

        # Prepare training data for fitting
        if self.custom_dir:
            train_data = train_data.shuffle(self.builder.info.splits["train"].num_examples)

        else:
            train_data = train_data.shuffle(ds_info.splits["train"].num_examples)

        if self.batch_size <= 0:
            print("Something is wrong with the batch size")
            return None

        else:
            train_data = train_data.cache()
            train_data = train_data.batch(self.batch_size)
            train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

        return train_data

    def test_data_prep(self, validation_or_test="validation"):
        """
        Method that generates independent and different training and validation datasets of normalised images and
        applies cache(), batch(), and prefetch().

        :return: A tuple containing the validation data and test data respectively.
        """
        #if self.custom_dir:
        #    test_data = self.builder.as_dataset(split="test", shuffle_files=True, as_supervised=True)

        #else:
        validation_data, val_info = tfds.load(self.name,
                                                split=validation_or_test + "[:50%]",
                                                shuffle_files=True,
                                                as_supervised=True,
                                                with_info=True)

        test_data, test_info = tfds.load(self.name, split=validation_or_test + "[-50%:]",
                                            shuffle_files=True,
                                            as_supervised=True,
                                            with_info=True)
        
        if self.norm_func:
            # Map the normalisation function
            validation_data = validation_data.map(self.norm_func,
                                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.resize:
            validation_data = validation_data.map(lambda image, label: (tf.image.resize(image, [self.resize[0],
                                                                                                self.resize[1]]),
                                                                        label))
        # Batch the validation set
        validation_data = validation_data.batch(self.batch_size)

        # Cache and Prefetch validation set
        validation_data = validation_data.cache()
        validation_data = validation_data.prefetch(tf.data.experimental.AUTOTUNE)

        if self.norm_func:
            # Map the normalisation function
            test_data = test_data.map(self.norm_func,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.resize:
            test_data = test_data.map(lambda image, label: (tf.image.resize(image, [self.resize[0], self.resize[1]]),
                                                            label))

        # Batch the test set
        test_data = test_data.batch(self.batch_size)

        # Cache and Prefetch sets
        test_data = test_data.cache()
        test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

        if self.custom_dir:
            return test_data

        else:
            return validation_data, test_data
