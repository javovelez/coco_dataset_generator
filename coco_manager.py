import copy
import shutil
import numpy as np
import os
import cv2
import itertools
import pandas as pd
import math

# image_template = '{{"id": {}, "file_name": "{}", "width": 1024, "height": 576, "date_captured": "", "license": 1, "coco_url": "", "flickr_url": ""}}'
# annotation_template = '{{"id": {an_id}, "image_id": {im_id}, "category_id": {cat}, "iscrowd": 0, "area": 0, "bbox": [{bb[0]}, {bb[1]}, {bb[2]}, {bb[3]}], "width": 1024, "height": 576, "center": [{center[0]}, {center[1]}],  "vertices": [{vertices[0]}, {vertices[1]}, {vertices[2]}, {vertices[3]}, {vertices[4]}, {vertices[5]}, {vertices[6]}, {vertices[7]}]}}'


class CocoManager:

    def __init__(self, images_path, xml_doc, images_perc, resize_factor,
                 dict_list, image_to_dict, annotation_to_dict
                 ):
        self.image_id = 1
        self.annotation_id = 1
        self.images_path = images_path
        self.xml_images_list = xml_doc.getElementsByTagName('image')
        self.images_perc = images_perc
        self.resize_factor = resize_factor
        self.dict_list_train = copy.deepcopy(dict_list)
        self.dict_list_test = copy.deepcopy(dict_list)
        self.image_to_dict = image_to_dict
        self.annotation_to_dict = annotation_to_dict
        self.prefix = ['', '']
        self.suffix = ['', '']
        self.rm = []
        self.n_images_train = None
        self.n_images_test = None
        self.labels = []

    def set_image_id(self, new_id):
        self.image_id = new_id

    def set_annotation_id(self, new_id):
        self.annotation_id = new_id

    def set_n_images_train(self, n):
        self.n_images_train = n

    def set_n_images_test(self,n):
        self.n_images_test = n

    def set_rm(self,rm):
        self.rm = rm

    def set_prefix(self, prefix):
        self.prefix = prefix

    def set_suffix(self, suffix):
        self.suffix = suffix

    def set_dict_list(self, dict, group, key):
        if group == 'train':
            self.dict_list_train[key].append(dict)
        elif group == 'test':
            self.dict_list_test[key].append(dict)
        else:
            raise Exception

    def image_name_parse(self, name):
        name = name.replace(self.prefix[0], self.prefix[1])
        name = name.replace(self.suffix[0], self.suffix[1])
        return name

    def get_centre_radius_circle_from_3_points(self, pts):
        x1 = pts[0]
        x2 = pts[2]
        x3 = pts[4]
        y1 = pts[1]
        y2 = pts[3]
        y3 = pts[5]

        x12 = x1 - x2
        x13 = x1 - x3

        y12 = y1 - y2
        y13 = y1 - y3

        y31 = y3 - y1
        y21 = y2 - y1

        x31 = x3 - x1
        x21 = x2 - x1

        # x1^2 - x3^2
        sx13 = pow(x1, 2) - pow(x3, 2)

        # y1^2 - y3^2
        sy13 = pow(y1, 2) - pow(y3, 2)

        sx21 = pow(x2, 2) - pow(x1, 2)
        sy21 = pow(y2, 2) - pow(y1, 2)

        f = (((sx13) * (x12) + (sy13) *
              (x12) + (sx21) * (x13) +
              (sy21) * (x13)) // (2 *
                                  ((y31) * (x12) - (y21) * (x13))))

        g = (((sx13) * (y12) + (sy13) * (y12) +
              (sx21) * (y13) + (sy21) * (y13)) //
             (2 * ((x31) * (y12) - (x21) * (y13))))

        c = (-pow(x1, 2) - pow(y1, 2) -
             2 * g * x1 - 2 * f * y1)

        # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
        # where centre is (h = -g, k = -f) and
        # radius r as r^2 = h^2 + k^2 - c
        h = -g
        k = -f

        sqr_of_r = h * h + k * k - c
        # r is the radius
        r = round(math.sqrt(sqr_of_r), 5)
        x = h
        y = k

        return x, y, r

    def bb_from_circle(self,center,radius):
        bb = [round(center[0] - radius), round(center[1] - radius), round(2 * radius), round(2 * radius)]
        bb = [int(x) for x in bb]
        return bb

    def draw_circle(self, img, center, radius, color=(23, 220, 75), thickness=1):
        img = cv2.circle(img, center, radius, color, thickness)

    def draw_polygons(self,img, vertices, is_closed=True, color=(23, 220, 75), thickness=1):
        vertices = np.array([[int(round(vertices[i])), int(round(vertices[i+1]))] for i in range(0, len(vertices)-1, 2)])
        cv2.polylines(img, [vertices], is_closed, color, thickness)

    def draw_points(self,img,  points_list, radius=1 ,color=(23, 220, 75), thickness=-1):
        for point in points_list:
            point = tuple( int(round(p)) for p in point)
            cv2.circle(img, point, radius, color, thickness=thickness)
    def get_train_dict(self, output_path='.', filename=''):
        return self.dict_list_train

    def get_test_dict(self, output_path='.', filename=''):
        return self.dict_list_test

    def inc_img_id(self):
        self.image_id += 1

    def inc_annotation_id(self):
        self.annotation_id += 1

    def save_images(self, dir_source, dir_target, group):
        # dict = self.dict_list_train if group == 'train' else self.dict_list_test
        # images_list = dict["images"]
        resize_factor = self.resize_factor
        df = self.get_labels_dataframe()
        df_group = df[df['group']==group]
        images_list = df_group["image_name"]
        image_list = list(pd.unique(images_list))

        if resize_factor != 1:
            for image_name in images_list:
                img = cv2.imread(dir_source + image_name)
                if img is None:
                    continue
                if resize_factor > 1:
                    print("Cuidado, está usando una interpolación para reducir tamaño pero lo está aumentando")
                if img.shape[0] > img.shape[1]:
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img = cv2.resize(img, None, interpolation=cv2.INTER_AREA, fx=resize_factor, fy=resize_factor)
                cv2.imwrite(dir_target + image_name, img)
        else:
            rotated_df = df_group[df_group['rotate']==True]

            images_rotated_list = list(pd.unique(rotated_df['image_name']))
            for image_to_rotate in images_rotated_list:
                img = cv2.imread(dir_source+ image_to_rotate)
                if img is None:
                    continue
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite(dir_target + image_to_rotate, img)
            not_rotated_df = df_group[df_group['rotate']==False]
            images_not_rotated_list = list(pd.unique(not_rotated_df['image_name']))
            for image_to_copy in images_not_rotated_list:
                if os.path.exists(dir_source + image_to_copy):
                    shutil.copyfile(dir_source + image_to_copy, dir_target + image_to_copy)

    def get_img_id(self):
        return self.img_id

    def get_annotation_id(self):
        return self.annotation_id

    def parse_to_coco(self):
        raise NotImplementedError

    def draw_labels(self, dir):
        raise NotImplementedError

    def get_labels_dataframe(self):
        raise NotImplementedError
