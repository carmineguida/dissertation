import sys
import math
import random
import csv
import numpy as np
import cv2

DATA_CSV_PATH = '../oai/xray_split_flip/data.csv'
IMAGE_FILE_PATH = '../oai/xray_split_flip/images/'
PTS_FILE_PATH = '../oai/xray_split_flip/pts/'

DIMETA_PATH = 'data/knee_dimeta.csv'

OUTPUT_FILE_PATH = 'data/xray_smartcrop/'

ID_COL = 'id'
SIDE_COL = 'side'
KL_COL = 'kl'

SPACING_X = 0.2
SPACING_Y = 0.2

BASE_SCALE_WIDTH = 1200
CENTER_POINT = 55
CENTER_POINT_OFFSET = [10, 0]
CROP_WIDTH = 600
CROP_HEIGHT = 220

def get_cols(line, c=','):
    line = line.replace('\n', '').replace('\"', '')
    cols = line.split(c)
    return cols

def load_dict_from_file(filepath):
    d = {}
    with open(filepath, 'r') as f:
        s = f.read()
        s = s.replace('null', 'None').replace('true', 'True').replace('false', 'False')
        d = eval(s)

    return d
    
def load_dimeta():
    dataset = {}

    f = open(DIMETA_PATH, 'r')
    line = f.readline()
    fields = get_cols(line)
    id_col = fields.index('case_id')
    spacing_col = fields.index('spacing_0')

    while True:
        line = f.readline()
        if (line is None):
            break
        cols = get_cols(line)
        if (len(cols) < len(fields)):
            break
        case_id = cols[id_col]
        spacing = cols[spacing_col]
        dataset[case_id] = float(spacing)
    f.close()
    return dataset

def load_dataset(dimeta):
    dataset = []
    
    f = open(DATA_CSV_PATH, 'r')
    line = f.readline()
    fields = get_cols(line)

    id_col = fields.index(ID_COL)
    side_col = fields.index(SIDE_COL)
    kl_col = fields.index(KL_COL)

    while True:
        line = f.readline()
        if (line is None):
            break
        cols = get_cols(line)
        if (len(cols) < len(fields)):
            break

        case_id = cols[id_col]
        side = cols[side_col]
        kl = cols[kl_col]
        
        data = {'case_id':case_id, 'side':side, 'kl':kl, 'spacing':dimeta[case_id]}

        full_id = case_id + '_' + side

        dataset.append(data)

    f.close()
    return dataset

######################################################################

def get_points(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    for i in range(3):
        lines.pop(0)
    lines.pop()
    points = []
    for line in lines:
        l = line.strip()
        p = l.split(' ')
        x = int(float(p[0]))
        y = int(float(p[1]))
        points.append((x, y))
    return points

def scale_image(image):
  (height, width) = image.shape
  new_width = BASE_SCALE_WIDTH
  ratio = float(new_width) / float(width)
  new_height = int(ratio * float(height))
  scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
  return scaled

def scale_points(original, scaled, points):
  adjusted = []

  (original_height, original_width) = original.shape
  (scaled_height, scaled_width) = scaled.shape

  height_ratio = float(scaled_height) / float(original_height)
  width_ratio = float(scaled_width) / float(original_width)

  for p in points:
    x = int(float(p[0]) * width_ratio)
    y = int(float(p[1]) * height_ratio)
    adjusted.append((x, y))

  return adjusted

def crop_center(scaled_image, scaled_points):
  shape = scaled_image.shape

  p = scaled_points[CENTER_POINT]
  
  x = p[0] + CENTER_POINT_OFFSET[0]
  y = p[1] + CENTER_POINT_OFFSET[1]

  width = CROP_WIDTH
  height = CROP_HEIGHT
  y -= int(CROP_HEIGHT / 2)
  if (y < 0):
      y = 0
  if (y + height >= shape[0]):
      y = shape[0] - height

  x -= int(CROP_WIDTH / 2)
  if (x < 0):
      x = 0
  if (x + width >= shape[1]):
      x = shape[1] - width

  return scaled_image[y:y + height, x:x + width]

######################################################################

def voxel_scale(image, points, spacing):
    scale_x = SPACING_X / spacing
    scale_y = SPACING_Y / spacing

    shape = image.shape
    width = int(float(shape[1]) * scale_x)
    height = int(float(shape[0]) * scale_y)

    scaled_image = cv2.resize(image, (width, height))
    scaled_points = scale_points(image, scaled_image, points)

    return (scaled_image, scaled_points)

def base_scale(image, points):
    scaled_image = scale_image(image)
    scaled_points = scale_points(image, scaled_image, points)
    return (scaled_image, scaled_points)

######################################################################

def save_set(dataset, count, path):
    error_count = 0

    for i in range(count):
        data = dataset.pop(0)

        case_id = data['case_id']
        side = data['side']
        kl = data['kl']
        spacing = data['spacing']

        prefix = case_id[1]
        image_path = IMAGE_FILE_PATH + prefix + '/' + case_id + '_' + side + '.png'
        pts_path = PTS_FILE_PATH + prefix + '/' + case_id + '_' + side + '_R.pts'

        image = cv2.imread(image_path, 0)
        points = get_points(pts_path)
        
        (image, points) = voxel_scale(image, points, spacing)

        try:
            (image, points) = base_scale(image, points)

            cropped = crop_center(image, points)

            output_path = OUTPUT_FILE_PATH + path + '/' + kl + '/' + case_id + '_' + side + '.png'
            cv2.imwrite(output_path, cropped)

        except:
            error_count += 1
            continue

    print('errors:', error_count)

######################################################################

dimeta = load_dimeta()

dataset = load_dataset(dimeta)
random.shuffle(dataset)

total = len(dataset)
count = int(total * 0.70)
save_set(dataset, count, 'train')

total = len(dataset)
count = int(total * 0.33)
save_set(dataset, count, 'val')

total = len(dataset)
count = total
save_set(dataset, count, 'test')


