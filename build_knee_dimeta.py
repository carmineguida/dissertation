import csv
import pydicom

################################################################################

XRAY_FILE_PATH = './OAIScreeningImages/enrollee01.txt'
XRAY_ID_COL = 'src_subject_id'
XRAY_IMAGE_TYPE = 'Bilateral PA Fixed Flexion Knee'
XRAY_IMAGE_PATH_PREFIX = 's3://NDAR_Central_2/submission_13305/00m/'
XRAY_CSV_OUTPUT_PATH = 'oai100xray.csv'

IMAGE_FILE_PATH = './OAINonImaging/image03.txt'
IMAGE_BASE_PATH = './OAIScreeningImages/results/'
IMAGE_ID_COL = 'src_subject_id'
IMAGE_TYPE_COL = 'image_description'
IMAGE_PATH_COL = 'image_file'
IMAGE_PATH_PREFIX = 's3://NDAR_Central_1/submission_13364/00m/'
IMAGE_GROUP_1 = '0.E.1'
IMAGE_GROUP_2 = '0.C.2'

USE_ID_COL = 'case_id'

OUTPUT_FILE = 'data/knee_dimeta.csv'

################################################################################

def check_int(s):
    try:
        return int(s)
    except:
        return -1

def clean_image_path(path, prefix):
    path = path.replace(prefix, '').replace('.tar.gz', '')
    return path

def get_cols(line, c='\t'):
    cols = line.split(c)
    return cols

################################################################################

def load_xray_dataset():
    dataset = {}
    f = open(XRAY_FILE_PATH, 'r')
    line = f.readline().replace('\n', '').replace('\"', '')
    fields = get_cols(line)
    id_col = fields.index(XRAY_ID_COL)

    # Skip next line
    f.readline()

    while True:
        line = f.readline()
        if (line is None):
            break

        cols = get_cols(line.replace('\n', '').replace('\"', ''))
        if (len(cols) < id_col):
            break

        xray_id = cols[id_col]
        if (xray_id in dataset):
            continue
        data = { 'id':xray_id }
        dataset[xray_id] = data

    f.close()
    return dataset

################################################################################

def load_dataset(xray_dataset):
    dataset = []

    f = open(IMAGE_FILE_PATH, 'r')

    # Get indexes of the columns we want
    line = f.readline().replace('\n', '').replace('\"', '')
    fields = get_cols(line)
    id_col = fields.index(IMAGE_ID_COL)
    type_col = fields.index(IMAGE_TYPE_COL)
    path_col = fields.index(IMAGE_PATH_COL)

    # Skip next line
    f.readline()

    while True:
        line = f.readline()
        if (line is None):
            break
        
        cols = get_cols(line.replace('\n', '').replace('\"', ''))
        if (len(cols) < type_col):
            break
        
        image_id = cols[id_col]

        if (image_id not in xray_dataset):
            continue

        image_type = cols[type_col]
        if (image_type != XRAY_IMAGE_TYPE):
            continue

        image_path = clean_image_path(cols[path_col], XRAY_IMAGE_PATH_PREFIX)
        if (image_path.startswith(IMAGE_GROUP_1) == False and image_path.startswith(IMAGE_GROUP_2) == False):
            continue

        data = { 'case_id':image_id, 'path':image_path }
        try:
            ds = pydicom.dcmread(IMAGE_BASE_PATH + image_path + '/001')
            spacing = ds.PixelSpacing
            data['spacing_0'] = spacing[0]
            data['spacing_1'] = spacing[1]
            dataset.append(data)
        except:
            print('Error with:', image_path)

    f.close()    
    return dataset

################################################################################

def write_csv(dataset):
    fieldnames = ['case_id', 'path', 'spacing_0', 'spacing_1']
    with open(OUTPUT_FILE, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in dataset:
            writer.writerow(row)
    
################################################################################

xray_dataset = load_xray_dataset()
dataset = load_dataset(xray_dataset)
write_csv(dataset)

