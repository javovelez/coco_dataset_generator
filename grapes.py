import copy
import random

import numpy
import pandas as pd
from coco_manager import CocoManager
import numpy as np
import itertools
import math
import json
import cv2


class GrapesMixTasks_159_268_Jobs_144_246(CocoManager):
    two_frames = False

    def set_two_frames(self):
        self.two_frames = True



    def parse_to_coco(self):
        resize_factor = self.resize_factor
        xml_image_list = self.xml_images_list
        for image in xml_image_list:
            points_tag_list = image.getElementsByTagName('points')
            img_xml_name = image.attributes["name"].value
            image_name = self.image_name_parse(img_xml_name)
            if len(points_tag_list) == 0:
                # print("WARNING: no se encontraron etiquetas para la imagen ", image_name)
                continue
            img_xml_id = image.attributes["id"].value

            if int(img_xml_id) > 104 or (image_name in self.rm):
                continue
            if self.two_frames:
                img_width = int(float(image.attributes["width"].value) * resize_factor / 2)
            else:
                img_width = int(float(image.attributes["width"].value) * resize_factor)

            img_height = int(float(image.attributes["height"].value) * resize_factor)

            if self.image_id <= self.n_images_train:
                group = 'train'
            else:
                group = 'test'
            rotate = True if (img_height > img_width) else False
            if rotate:
                img_width, img_height = img_height, img_width
            image_dict = self.image_to_dict(self.image_id, image_name, int(img_width), int(img_height))

            self.set_dict_list(image_dict, group, 'images')

            for points_attribute in points_tag_list:
                label = points_attribute.attributes['label'].value
                if label != 'baya':
                    continue
                try:
                    group_id = points_attribute.attributes['group_id'].value
                except:
                    group_id = '-'
                    # print(f'No se encuentra group_id en una etiqueta de la imagen {image_name} img_id = {img_xml_id}')
                points_str = points_attribute.attributes['points'].value
                cat = 1  # points_attribute.attributes['label'].value
                points = [list(map(float, x.split(','))) for x in points_str.split(';')]
                points_to_delete = []
                for point_idx, point in enumerate(points):
                    point[0] = point[0] * resize_factor
                    point[1] = point[1] * resize_factor
                    if rotate:
                        point[0], point[1] = point[1], img_height - point[0]
                    if point[0] > img_width or point[1] > img_height or point[1] < 0:
                        points_to_delete.append(point_idx)
                for idx in sorted(points_to_delete, reverse=True):
                    points.pop(idx)
                if len(points) != 3:
                    print(f"Error de etiquetado en imagen: {image_name}, img_id = {img_xml_id}, "
                          f"group_id = {group_id}, tiene {len(points)} puntos")
                    continue
                points = list(itertools.chain(*points))
                x_center, y_center, radius = self.get_centre_radius_circle_from_3_points(points)

                bb = self.bb_from_circle([x_center, y_center], radius)
                center = [x_center, y_center]
                radius = radius
                annotation_dict = self.annotation_to_dict(self.annotation_id, self.image_id, cat, bb, img_width,
                                                          img_height, center, radius)
                self.set_dict_list(annotation_dict, group, 'annotations')
                self.labels.append(
                    (image_name, center, radius, bb, cat, rotate, group, self.annotation_id, self.image_id))
                self.inc_annotation_id()
            self.inc_img_id()

    def get_labels_dataframe(self):  # usado para rotar y escalar imágenes
        df = []
        for label in self.labels:
            df.append([label[0], label[1][0], label[1][1], label[2], label[4], label[5], label[6], label[7], label[8]])
        df = pd.DataFrame(df, columns=["image_name", "center_x", "center_y", "radius", "cat", "rotate", "group",
                                       "annotation_id", "image_id"])
        df = df.astype({'image_name': str, 'center_x': int, 'center_y': int, 'radius': int, 'cat': int,
                        'rotate': bool, 'group': str, 'image_id': int})
        return df

    def draw_labels(self, json_path, images_dir, output_dir):
        f = open(json_path)
        data = json.load(f)
        images_list = data['images']
        annotations_list = data['annotations']
        images_dict = {}
        annotation_list = []
        for image_dict in images_list:
            images_dict[image_dict['id']] = image_dict['file_name']

        for annotations_dict in annotations_list:
            lista = [annotations_dict['image_id'], annotations_dict['circle_center'][0],
                     annotations_dict['circle_center'][1], annotations_dict['circle_radius']]
            annotation_list.append(lista)
        annotations_df = pd.DataFrame(annotation_list, columns=["image_id", 'center_x', 'center_y', 'radius'])
        annotations_ids = annotations_df["image_id"]
        for img_id, image_name in images_dict.items():
            image_path = images_dir + image_name
            img = cv2.imread(image_path)
            ann_df = annotations_df[annotations_df['image_id'] == img_id]
            for idx in range(len(ann_df)):
                image_id, center_x, center_y, radius = list(ann_df.iloc[idx])
                image_id, center_x, center_y, radius = int(round(image_id)),   int(round(center_x)),  int(round(center_y)),  int(round(radius))
                self.draw_circle(img, (center_x, center_y), radius)
            print(output_dir + image_name)
            cv2.imwrite(output_dir + image_name, img)

    def draw_labels_debug(self, json_path, images_dir, output_dir):
        f = open(json_path)
        data = json.load(f)
        images_list = data['images']
        annotations_list = data['annotations']
        images_dict = {}
        annotation_list = []
        for image_dict in images_list:
            images_dict[image_dict['id']] = image_dict['file_name']

        for annotations_dict in annotations_list:
            lista = [annotations_dict['image_id'], annotations_dict['circle_center'][0],
                     annotations_dict['circle_center'][1], annotations_dict['circle_radius']]
            annotation_list.append(lista)
        annotations_df = pd.DataFrame(annotation_list, columns=["image_id", 'center_x', 'center_y', 'radius'])
        annotations_ids = annotations_df["image_id"]
        for img_id, image_name in images_dict.items():
            image_path = images_dir + image_name
            img = cv2.imread(image_path)
            if img is None:
                continue
            if img.shape[0] > img.shape[1]:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ann_df = annotations_df[annotations_df['image_id'] == img_id]
            for idx in range(len(ann_df)):
                image_id, center_x, center_y, radius = list(ann_df.iloc[idx])
                image_id, center_x, center_y, radius = int(round(image_id)), int(round(center_x)), int(round(center_y)), int(round(radius))
                self.draw_circle(img, (center_x, center_y), radius)
            print(output_dir + image_name)
            cv2.imwrite(output_dir + image_name, img)

    def get_occlusion_factor(self, points, radius):
        v1_x = points[0]
        v1_y = points[1]
        v2_x = points[2]
        v2_y = points[3]
        v3_x = points[4]
        v3_y = points[5]

        v_21x = v2_x - v1_x
        v_21y = v2_y - v1_y

        v_32x = v3_x - v2_x
        v_32y = v3_y - v2_y

        v_13x = v1_x - v3_x
        v_13y = v1_y - v3_y

        v1_mod = (v_21x**2 + v_21y**2)**0.5
        v2_mod = (v_32x**2 + v_32y**2)**0.5
        v3_mod = (v_13x**2 + v_13y**2)**0.5

        max_area_triangle = 2 * radius**2 * (np.cos(np.pi/6))**3

        s_p = (v1_mod + v2_mod + v3_mod) / 2  # semi perímetro
        triangle_area = (s_p * (s_p - v1_mod) * (s_p - v2_mod) * (s_p - v3_mod)) ** 0.5
        circle_area = np.pi * radius ** 2

        # print(f"radius: {radius}")
        # print(f"max_area_triangle: {max_area_triangle}")
        # print(f"triangle_area: {triangle_area}")
        # print(f"s_p: {s_p}")
        # print(f"v1_mod: {v1_mod}")
        # print(f"v2_mod: {v2_mod}")
        # print(f"v3_mod: {v3_mod}")
        # print(f"circle_area: {circle_area}")
        # print(triangle_area / max_area_triangle)
        # if triangle_area / max_area_triangle > 1:
        #     input()

        return triangle_area / max_area_triangle


class GrapesMixTasksIOU(GrapesMixTasks_159_268_Jobs_144_246):
    def parse_to_coco(self):
        resize_factor = self.resize_factor
        xml_image_list = self.xml_images_list
        for image in xml_image_list:
            points_tag_list = image.getElementsByTagName('points')
            img_xml_name = image.attributes["name"].value
            image_name = self.image_name_parse(img_xml_name)
            if len(points_tag_list) == 0:
                # print("WARNING: no se encontraron etiquetas para la imagen ", image_name)
                continue
            img_xml_id = image.attributes["id"].value

            if int(img_xml_id) > 104 or (image_name in self.rm):
                continue
            if self.two_frames:
                img_width = int(float(image.attributes["width"].value) * resize_factor / 2)
            else:
                img_width = int(float(image.attributes["width"].value) * resize_factor)

            img_height = int(float(image.attributes["height"].value) * resize_factor)

            if self.image_id <= self.n_images_train:
                group = 'train'
            else:
                group = 'test'
            rotate = True if (img_height > img_width) else False
            if rotate:
                img_width, img_height = img_height, img_width
            image_dict = self.image_to_dict(self.image_id, image_name, int(img_width), int(img_height))

            self.set_dict_list(image_dict, group, 'images')

            for points_attribute in points_tag_list:
                label = points_attribute.attributes['label'].value
                if label != 'baya':
                    continue
                try:
                    group_id = points_attribute.attributes['group_id'].value
                except:
                    group_id = '-'
                    # print(f'No se encuentra group_id en una etiqueta de la imagen {image_name} img_id = {img_xml_id}')
                points_str = points_attribute.attributes['points'].value
                cat = 1  # points_attribute.attributes['label'].value
                points = [list(map(float, x.split(','))) for x in points_str.split(';')]
                points_to_delete = []
                for point_idx, point in enumerate(points):
                    point[0] = point[0] * resize_factor
                    point[1] = point[1] * resize_factor
                    if rotate:
                        point[0], point[1] = point[1], img_height - point[0]
                    if point[0] > img_width or point[1] > img_height or point[1] < 0:
                        points_to_delete.append(point_idx)
                for idx in sorted(points_to_delete, reverse=True):
                    points.pop(idx)
                if len(points) != 3:
                    print(f"Error de etiquetado en imagen: {image_name}, img_id = {img_xml_id}, "
                          f"group_id = {group_id}, tiene {len(points)} puntos")
                    continue
                points_to_annotate = copy.deepcopy(points)
                points = list(itertools.chain(*points))
                x_center, y_center, radius = self.get_centre_radius_circle_from_3_points(points)
                occlusion_factor = self.get_occlusion_factor(points, radius)
                bb = self.bb_from_circle([x_center, y_center], radius)
                center = [x_center, y_center]


                annotation_dict = self.annotation_to_dict(self.annotation_id, self.image_id, cat, bb, img_width,
                                                          img_height, center, radius, occlusion_factor, points_to_annotate)
                self.set_dict_list(annotation_dict, group, 'annotations')
                self.labels.append(
                    (image_name, center, radius, bb, cat, rotate, group, self.annotation_id, self.image_id))
                self.inc_annotation_id()
            self.inc_img_id()

    def draw_labels(self, json_path, images_dir, output_dir):
        f = open(json_path)
        data = json.load(f)
        images_list = data['images']
        annotations_list = data['annotations']
        images_dict = {}
        annotation_list = []
        for image_dict in images_list:
            images_dict[image_dict['id']] = image_dict['file_name']

        for annotations_dict in annotations_list:
            lista = [annotations_dict['image_id'], annotations_dict['circle_center'][0],
                     annotations_dict['circle_center'][1], annotations_dict['circle_radius'],
                     annotations_dict['occlusion_factor'], annotations_dict['points']]
            annotation_list.append(lista)
        annotations_df = pd.DataFrame(annotation_list, columns=["image_id", 'center_x', 'center_y', 'radius', 'occlusion_factor', 'points'])
        annotations_df.astype({"image_id": int, 'center_x': int, 'center_y': int, 'radius': int, 'occlusion_factor': float, 'points': str})
        for img_id, image_name in images_dict.items():
            image_path = images_dir + image_name
            img = cv2.imread(image_path)
            if img is None:
                continue
            if img.shape[0] > img.shape[1]:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ann_df = annotations_df[annotations_df['image_id'] == img_id]
            for idx in range(len(ann_df)):
                image_id, center_x, center_y, radius, occlusion_factor, points = list(ann_df.iloc[idx])
                image_id, center_x, center_y, radius = int(round(image_id)), int(round(center_x)), int(round(center_y)), int(round(radius))
                occlusion_factor = round(occlusion_factor, 2)
                # points = [list(map(float, x.split(','))) for x in points.split(';')]
                for point_idx, point in enumerate(points):
                    for coor_idx, coor in enumerate(point):
                        points[point_idx][coor_idx] = int(round(coor))
                random_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

                self.draw_circle(img, (center_x, center_y), radius, color=random_color)
                cv2.polylines(img, [numpy.asarray(points)], True, random_color,thickness=1)

                # draw occlusion's factors
                org = (center_x, center_y)
                fontScale = 0.3
                color = (0, 255, 0)
                thickness = 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(occlusion_factor), org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
            print(output_dir + image_name)
            cv2.imwrite(output_dir + image_name, img)