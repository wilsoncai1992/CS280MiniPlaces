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
import cPickle as pickle
import gzip


## GLOBAL VARIABLES ##
global IMG_SIZE
IMG_SIZE = 224
# IMG_SIZE = 128

# TODO: dynamic is better
global LABELS_DICT
LABELS_DICT = {
        'Cani': 3,
        'Cavalli': 2,
        'Alberi': 1,
        'Gatti': 0,

        # 'abbey': 0,
        # 'airport_terminal': 1,
        # 'amphitheater': 2,
        # 'amusement_park': 3,
        # 'aquarium': 4,
        # 'aqueduct': 5,
        # 'art_gallery': 6,
        # 'assembly_line': 7,
        # 'auditorium': 8,
        # 'badlands': 9,
        # 'bakery/shop': 10,
        # 'ballroom': 11,
        # 'bamboo_forest': 12,
        # 'banquet_hall': 13,
        # 'bar': 14,
        # 'baseball_field': 15,
        # 'bathroom': 16,
        # 'beauty_salon': 17,
        # 'bedroom': 18,
        # 'boat_deck': 19,
        # 'bookstore': 20,
        # 'botanical_garden': 21,
        # 'bowling_alley': 22,
        # 'boxing_ring': 23,
        # 'bridge': 24,
        # 'bus_interior': 25,
        # 'butchers_shop': 26,
        # 'campsite': 27,
        # 'candy_store': 28,
        # 'canyon': 29,
        # 'cemetery': 30,
        # 'chalet': 31,
        # 'church/outdoor': 32,
        # 'classroom': 33,
        # 'clothing_store': 34,
        # 'coast': 35,
        # 'cockpit': 36,
        # 'coffee_shop': 37,
        # 'conference_room': 38,
        # 'construction_site': 39,
        # 'corn_field': 40,
        # 'corridor': 41,
        # 'courtyard': 42,
        # 'dam': 43,
        # 'desert/sand': 44,
        # 'dining_room': 45,
        # 'driveway': 46,
        # 'fire_station': 47,
        # 'food_court': 48,
        # 'fountain': 49,
        # 'gas_station': 50,
        # 'golf_course': 51,
        # 'harbor': 52,
        # 'highway': 53,
        # 'hospital_room': 54,
        # 'hot_spring': 55,
        # 'ice_skating_rink/outdoor': 56,
        # 'iceberg': 57,
        # 'kindergarden_classroom': 58,
        # 'kitchen': 59,
        # 'laundromat': 60,
        # 'lighthouse': 61,
        # 'living_room': 62,
        # 'lobby': 63,
        # 'locker_room': 64,
        # 'market/outdoor': 65,
        # 'martial_arts_gym': 66,
        # 'monastery/outdoor': 67,
        # 'mountain': 68,
        # 'museum/indoor': 69,
        # 'office': 70,
        # 'palace': 71,
        # 'parking_lot': 72,
        # 'phone_booth': 73,
        # 'playground': 74,
        # 'racecourse': 75,
        # 'railroad_track': 76,
        # 'rainforest': 77,
        # 'restaurant': 78,
        # 'river': 79,
        # 'rock_arch': 80,
        # 'runway': 81,
        # 'shed': 82,
        # 'shower': 83,
        # 'ski_slope': 84,
        # 'skyscraper': 85,
        # 'slum': 86,
        # 'stadium/football': 87,
        # 'stage/indoor': 88,
        # 'staircase': 89,
        # 'subway_station/platform': 90,
        # 'supermarket': 91,
        # 'swamp': 92,
        # 'swimming_pool/outdoor': 93,
        # 'temple/east_asia': 94,
        # 'track/outdoor': 95,
        # 'trench': 96,
        # 'valley': 97,
        # 'volcano': 98,
        # 'yard': 99,

        # '/a/abbey': 0,
        # '/a/airport_terminal': 1,
        # '/a/amphitheater': 2,
        # '/a/amusement_park': 3,
        # '/a/aquarium': 4,
        # '/a/aqueduct': 5,
        # '/a/art_gallery': 6,
        # '/a/assembly_line': 7,
        # '/a/auditorium': 8,
        # '/b/badlands': 9,
        # '/b/bakery/shop': 10,
        # '/b/ballroom': 11,
        # '/b/bamboo_forest': 12,
        # '/b/banquet_hall': 13,
        # '/b/bar': 14,
        # '/b/baseball_field': 15,
        # '/b/bathroom': 16,
        # '/b/beauty_salon': 17,
        # '/b/bedroom': 18,
        # '/b/boat_deck': 19,
        # '/b/bookstore': 20,
        # '/b/botanical_garden': 21,
        # '/b/bowling_alley': 22,
        # '/b/boxing_ring': 23,
        # '/b/bridge': 24,
        # '/b/bus_interior': 25,
        # '/b/butchers_shop': 26,
        # '/c/campsite': 27,
        # '/c/candy_store': 28,
        # '/c/canyon': 29,
        # '/c/cemetery': 30,
        # '/c/chalet': 31,
        # '/c/church/outdoor': 32,
        # '/c/classroom': 33,
        # '/c/clothing_store': 34,
        # '/c/coast': 35,
        # '/c/cockpit': 36,
        # '/c/coffee_shop': 37,
        # '/c/conference_room': 38,
        # '/c/construction_site': 39,
        # '/c/corn_field': 40,
        # '/c/corridor': 41,
        # '/c/courtyard': 42,
        # '/d/dam': 43,
        # '/d/desert/sand': 44,
        # '/d/dining_room': 45,
        # '/d/driveway': 46,
        # '/f/fire_station': 47,
        # '/f/food_court': 48,
        # '/f/fountain': 49,
        # '/g/gas_station': 50,
        # '/g/golf_course': 51,
        # '/h/harbor': 52,
        # '/h/highway': 53,
        # '/h/hospital_room': 54,
        # '/h/hot_spring': 55,
        # '/i/ice_skating_rink/outdoor': 56,
        # '/i/iceberg': 57,
        # '/k/kindergarden_classroom': 58,
        # '/k/kitchen': 59,
        # '/l/laundromat': 60,
        # '/l/lighthouse': 61,
        # '/l/living_room': 62,
        # '/l/lobby': 63,
        # '/l/locker_room': 64,
        # '/m/market/outdoor': 65,
        # '/m/martial_arts_gym': 66,
        # '/m/monastery/outdoor': 67,
        # '/m/mountain': 68,
        # '/m/museum/indoor': 69,
        # '/o/office': 70,
        # '/p/palace': 71,
        # '/p/parking_lot': 72,
        # '/p/phone_booth': 73,
        # '/p/playground': 74,
        # '/r/racecourse': 75,
        # '/r/railroad_track': 76,
        # '/r/rainforest': 77,
        # '/r/restaurant': 78,
        # '/r/river': 79,
        # '/r/rock_arch': 80,
        # '/r/runway': 81,
        # '/s/shed': 82,
        # '/s/shower': 83,
        # '/s/ski_slope': 84,
        # '/s/skyscraper': 85,
        # '/s/slum': 86,
        # '/s/stadium/football': 87,
        # '/s/stage/indoor': 88,
        # '/s/staircase': 89,
        # '/s/subway_station/platform': 90,
        # '/s/supermarket': 91,
        # '/s/swamp': 92,
        # '/s/swimming_pool/outdoor': 93,
        # '/t/temple/east_asia': 94,
        # '/t/track/outdoor': 95,
        # '/t/trench': 96,
        # '/v/valley': 97,
        # '/v/volcano': 98,
        # '/y/yard': 99,
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
