#it2.py
import json
import argparse
import funcy
import os, shutil

from sklearn.model_selection import train_test_split
'''
#train_dir = '/home/data/130/train_split.json'
#os.makedirs('train2017', exist_ok=True)
#test_dir = '/home/data/130/test_split.json'
#os.makedirs('test2017', exist_ok=True)
parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('annotations', metavar='coco_annotations', type=str,
                    help='Path to COCO annotations file.')
parser.add_argument('train', type=str, help='Where to store COCO training annotations')
parser.add_argument('test', type=str, help='Where to store COCO test annotations')
parser.add_argument('-s', dest='split', type=float, required=True,
                    help="A percentage of a split; a number in (0, 1)")

parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                    help='Ignore all images without annotations. Keep only these with at least one annotation')
parser.add_argument('images', type=str, help='Where images(dataset) is stored')


args = parser.parse_args()
'''

def save_coco(filep, info, licenses, images, annotations, categories):
    with open(filep, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images,
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def cocosplit(annotations = '/home/data/130/train.json', split = 0.9, train = '/home/data/130/train_split.json' , test = '/home/data/130/test_split.json'):
    with open(annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = 'None', #coco['info']
        licenses = 'None', #coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        #number_of_images = len(images)

        #images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        #if args.having_annotations:
        #    images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        x, y = train_test_split(images, train_size=split, random_state=42)
        #save_train = [shutil.copyfile(args.images + '/' + i['file_name'], train_dir + '/' + i['file_name']) for i in x]
        #save_test = [shutil.copyfile(args.images + '/' + j['file_name'], test_dir + '/' + j['file_name']) for j in y]
        save_coco(train, info, licenses, x, filter_annotations(annotations, x), categories)
        save_coco(test, info, licenses, y, filter_annotations(annotations, y), categories)

        print("Saved {} entries in {} and {} in {}".format(len(x), train, len(y), test))


#if __name__ == "__main__":
#    cocosplit(args.annotations, args.split, args.train, args.test)
