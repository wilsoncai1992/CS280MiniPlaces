#!/usr/bin/python
# -*- coding: utf-8 -*-

# ======================= DATASET ====================================
#
# Categories:
# 3) Cani
# 2) Cavalli
# 1) Alberi
# 0) Gatti
#
# ====================================================================
import glob
import os.path
import sys
import numpy as np
import tensorflow as tf
import logging as log
from timeit import default_timer as timer
try:
    import cPickle as pickle
except:
    import _pickle as pickle
import gzip


## GLOBAL VARIABLES ##
global IMG_SIZE
IMG_SIZE = 224
#IMG_SIZE = 128

# TODO: dynamic is better
global LABELS_DICT
LABELS_DICT = {
        'abbey': 0,
        'airport_terminal': 1,
        'amphitheater': 2,
        'amusement_park': 3,
        'aquarium': 4,
        'aqueduct': 5,
        'art_gallery': 6,
        'assembly_line': 7,
        'auditorium': 8,
        'badlands': 9,
        'bakery_shop': 10,
        'ballroom': 11,
        'bamboo_forest': 12,
        'banquet_hall': 13,
        'bar': 14,
        'baseball_field': 15,
        'bathroom': 16,
        'beauty_salon': 17,
        'bedroom': 18,
        'boat_deck': 19,
        'bookstore': 20,
        'botanical_garden': 21,
        'bowling_alley': 22,
        'boxing_ring': 23,
        'bridge': 24,
        'bus_interior': 25,
        'butchers_shop': 26,
        'campsite': 27,
        'candy_store': 28,
        'canyon': 29,
        'cemetery': 30,
        'chalet': 31,
        'church_outdoor': 32,
        'classroom': 33,
        'clothing_store': 34,
        'coast': 35,
        'cockpit': 36,
        'coffee_shop': 37,
        'conference_room': 38,
        'construction_site': 39,
        'corn_field': 40,
        'corridor': 41,
        'courtyard': 42,
        'dam': 43,
        'desert_sand': 44,
        'dining_room': 45,
        'driveway': 46,
        'fire_station': 47,
        'food_court': 48,
        'fountain': 49,
        'gas_station': 50,
        'golf_course': 51,
        'harbor': 52,
        'highway': 53,
        'hospital_room': 54,
        'hot_spring': 55,
        'ice_skating_rink_outdoor': 56,
        'iceberg': 57,
        'kindergarden_classroom': 58,
        'kitchen': 59,
        'laundromat': 60,
        'lighthouse': 61,
        'living_room': 62,
        'lobby': 63,
        'locker_room': 64,
        'market_outdoor': 65,
        'martial_arts_gym': 66,
        'monastery_outdoor': 67,
        'mountain': 68,
        'museum_indoor': 69,
        'office': 70,
        'palace': 71,
        'parking_lot': 72,
        'phone_booth': 73,
        'playground': 74,
        'racecourse': 75,
        'railroad_track': 76,
        'rainforest': 77,
        'restaurant': 78,
        'river': 79,
        'rock_arch': 80,
        'runway': 81,
        'shed': 82,
        'shower': 83,
        'ski_slope': 84,
        'skyscraper': 85,
        'slum': 86,
        'stadium_football': 87,
        'stage_indoor': 88,
        'staircase': 89,
        'subway_station_platform': 90,
        'supermarket': 91,
        'swamp': 92,
        'swimming_pool_outdoor': 93,
        'temple_east_asia': 94,
        'track_outdoor': 95,
        'trench': 96,
        'valley': 97,
        'volcano': 98,
        'yard': 99,
}


"""
Count total number of images
"""
def getNumImages(image_dir):
    count = 0
    for dirName, subdirList, fileList in os.walk(image_dir):
        for img in fileList:
            count += 1
    return count


"""
Return the dataset as images and labels
"""
def convertDataset(image_dir):

    num_labels = len(LABELS_DICT)
    label = np.eye(num_labels)  # Convert labels to one-hot-vector
    i = 0

    session = tf.Session()
    init = tf.initialize_all_variables()
    session.run(init)

    log.info("Start processing images (Dataset.py) ")
    start = timer()
    for dirName in os.listdir(image_dir):
        label_i = label[i]
        print("ONE_HOT_ROW = ", label_i)
        i += 1
        # log.info("Execution time of convLabels function = %.4f sec" % (end1-start1))
        path = os.path.join(image_dir, dirName)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            if os.path.isfile(img_path) and (img.endswith('jpeg') or
                                             (img.endswith('jpg'))):
                img_bytes = tf.read_file(img_path)
                img_u8 = tf.image.decode_jpeg(img_bytes, channels=3)
                img_u8_eval = session.run(img_u8)
                image = tf.image.convert_image_dtype(img_u8_eval, tf.float32)
                img_padded_or_cropped = tf.image.resize_image_with_crop_or_pad(image, IMG_SIZE, IMG_SIZE)
                img_padded_or_cropped = tf.reshape(img_padded_or_cropped, shape=[IMG_SIZE * IMG_SIZE, 3])
                yield img_padded_or_cropped.eval(session=session), label_i
                # img_padded_or_cropped.eval(session=session), label_i
    end = timer()
    log.info("End processing images (Dataset.py) - Time = %.2f sec" % (end-start))


def saveDataset(image_dir, file_path):
    with gzip.open(file_path, 'wb') as file:
        for img, label in convertDataset(image_dir):
            pickle.dump((img, label), file)


def loadDataset(file_path):
    with gzip.open(file_path) as file:
        while True:
            try:
                yield pickle.load(file)
            except EOFError:
                break
