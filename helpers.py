import shutil
from fpdf import FPDF
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
import random
import matplotlib.image as mpimg
import yaml


def normalize_img(image, label):
    """
    Function to map across all the images in a dataset in order to normalise them (e.g. normalise pixel values and
    set the datatype to float32).

    :param image: Image to normalise.
    :type image: array
    :param label: Target label of the image.
    :type label: int

    :return: Normalised image array and its label as a tuple.
    """
    
    new_img = tf.cast(image, tf.float32) / 255.
        
    return new_img, label


def mask_to_categorical(image, mask):
    image, mask = normalize_img(image, mask)
    mask = tf.one_hot(tf.cast(mask, tf.int32), 10)
    mask = tf.cast(mask, tf.float32)
    
    return image, mask


def monoExp(x, m, t, b):
    """
    Mapping general exponential function.

    :param x: Coefficient x.
    :type x: float
    :param m: Coefficient m.
    :type m: float
    :param t: Coefficient t.
    :type t: float
    :param b: Coefficient b.
    :type b: float

    :return: A mapped value into an exponential space determined by the input coefficients.
    """
    return m * np.exp(-t * x) + b


def powerlaw(x, m, t, b):
    """
    Mapping general power law function.

    :param x: Coefficient x.
    :type x: float
    :param m: Coefficient m.
    :type m: float
    :param t: Coefficient t.
    :type t: float
    :param b: Coefficient b.
    :type b: float

    :return: A mapped value into a power law space determined by the input coefficients.
    """
    return m * x ** (-t) + b


def plot_random_sample(dataset):
    """
    Function that plots a random image from the input dataset.

    :param dataset: Tensorflow dataset containing images from which the random image will be sampled.
    :type dataset: tf.data.Dataset

    :return: Corresponding plot.
    """
    random_index = np.random.randint(0, len([i for i in dataset.take(1)][0][0]))

    if type(dataset) == tuple:
        plt.imshow([i for i in dataset[0].take(1)][0][random_index].numpy())

    else:
        plt.imshow([i for i in dataset.take(1)][0][0][random_index].numpy())


def plot_fits(data_index_array, loss_array, params: list, power_or_exp: str, experiment_number,
              save="/common/users/ap19121/dynamics-of-inference-and-learning/results_figs/"):
    """
    Plotting function that will plot the fitted curve (either exponential or power law) using a set of coefficients.

    :param data_index_array: Data on the horizontal axis of the plot. This should correspond to an array containing the
    number of data samples at each training iteration.
    :type data_index_array: array
    :param loss_array: Array containing the loss at the end of each training iteration (vertical axis for our fitted
    line).
    :type loss_array: array
    :param params: List of parameters to fit the data.
    :type params: list
    :param power_or_exp: Parameter that determines wheter the data will be fitted with an exponential ("exp") or a power
    law curve.
    :type power_or_exp: str
    :param experiment_number: Number of the experiment.
    :type experiment_number: int
    :param save: If not False, directory name to save the figures displaying the results of the experiments.

    :return: Corresponding plot.
    """
    
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_-%H_%M_%S")
    
    # plot the results
    plt.figure()
    plt.plot(data_index_array, loss_array, '.', label="data")

    if power_or_exp == "exp":
        plt.plot(data_index_array, monoExp(data_index_array, params[0], params[1], params[2]), '--',
                 label="fitted exponential", color='green')

    else:
        plt.plot(data_index_array, powerlaw(data_index_array, params[0], params[1], params[2]), '--',
                 label="fitted powerlaw", color='red')

    plt.legend()
    plt.title("Fitted Curve")
    
    if save:
        plt.savefig(save + dt_string + f"_experiment_{experiment_number}")


def img_to_pdf(sdir:str, out_name:str, output_path=False):
    """
    Helper function to put together all of the images in a folder into a single pdf file.
    
    :param sdir: Directory where the target images are allocated.
    :type sdir: str
    :param out_name: Desired filename for the output pdf document.
    :type out_name: str
    :param output_path: Boolean parameter. If True, the function will print the output path.
    :type output_path: bool
    
    :return: None
    """
    
    pdf = FPDF()
    pdf.set_auto_page_break(0)
    
    img_list = [x for x in os.listdir(sdir) if x[0] != "."]
    
    for img in img_list:
        pdf.add_page()
        pdf.image(sdir+img)
    
    pdf.output(out_name+".pdf")
    
    if output_path:
        print(os.getcwd())


def get_labels(label_path):
    """
    Accepts a label path (in the form of a JSON) and returns the file
    as a Python object.
    """
    with open(label_path) as f:
        return json.load(f)


def copy_images(parent_folder, new_subset, original_path, dataset, labels_path, target_labels):
    """
    Copies images from a set of target classes into a new folder in order to create a sub-dataset from an already
    existing dataset.

    :param parent_folder: original folder where the dataset is located and the new dataset will be allocated.
    :type parent_folder: str
    :param new_subset: name of the new folder where the subset will be allocated within parent_folder.
    :type new_subset: str
    :param original_path: path inside the parent_folder where the parent dataset is located.
    :type original_path: str
    :param dataset: Train, test, validation, ...
    :type dataset: list
    :param labels_path: path where classes.txt is allocated (normally .../meta)
    :type labels_path: str
    :param target_labels: List of target labels to copy.
    :type target_labels: list
    """

    print(f"\nUsing {dataset} labels...")
    labels = get_labels(labels_path + dataset + ".json")

    for i in target_labels:
        # Make target directory
        os.makedirs(parent_folder + "/" + new_subset + "/" + dataset + "/" + i, exist_ok=True)

        # Get the target classes
        moved_img = []
        for j in labels[i]:
            og_path = parent_folder + original_path + j + ".jpg"
            new_path = parent_folder + "/" + new_subset + "/" + dataset + "/" + j + ".jpg"

            # Copy images from the original path to our new path
            shutil.copy2(og_path, new_path)
            moved_img.append(new_path)

        print(f"Copied {len(moved_img)} images from {dataset} dataset {i} class...")


def view_three_images(target_dir, target_class):
    """
    Selects three random image  from target_class in target_dir.

    Requires target_dir to be in format:
        target_dir
                 |target_class_1
                 |target_class_2
                 |...
    """
    target_path = target_dir + target_class
    filenames = os.listdir(target_path)
    target_images = random.sample(filenames, 3)

    # Plot images
    plt.figure(figsize=(15, 6))
    for i, img in enumerate(target_images):
        img_path = target_path + "/" + img
        plt.subplot(1, 3, i + 1)
        plt.imshow(mpimg.imread(img_path))
        plt.title(target_class)
        plt.axis("off")


def pickles_to_csv(folder_dir, target_folder):
    """
    folder_dir should contain just pickle files.
    """
    filenames = next(os.walk(folder_dir), (None, None, []))[2]  # [] if no file

    df_final = pd.DataFrame([])

    for file in filenames:
        df = pd.read_pickle(folder_dir + file)
        df_final = pd.concat((df_final, df))

    df_final.to_csv(target_folder)


def yaml_loader(filepath):
    """Loads a yaml file"""
    with open(filepath, "r") as file_descriptor:
        data = yaml.load(file_descriptor, Loader=yaml.Loader)
    
    return data


def yaml_dump(filepath, data):
    """Dumps data to a yaml file"""
    with open(filepath, "w") as file_descriptor:
        yaml.dump(data, file_descriptor)
