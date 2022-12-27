from coco_manager import *
import numpy as np
import json

class Quadrilateral(CocoManager):



    def get_category(self, cat):
        raise NotImplementedError

    def parse_to_coco(self):
        resize_factor = self.resize_factor
        xml_image_list = self.xml_images_list

        for img_number, image in enumerate(xml_image_list):
            points_tag_list = image.getElementsByTagName('polygon')
            img_xml_name = image.attributes["name"].value
            image_name = self.image_name_parse(img_xml_name)

            if len(points_tag_list) == 0:
                # print("WARNING: no se encontraron etiquetas para la imagen ", image_name)
                continue
            img_xml_id = image.attributes["id"].value

            img_width = int(float(image.attributes["width"].value) * resize_factor)
            img_height = int(float(image.attributes["height"].value) * resize_factor)

            if img_number < self.n_images_train:
                group = 'train'
            else:
                group = 'test'

            rotate = True if (img_height > img_width) else False
            if rotate:
                img_width, img_height = img_height, img_width

            image_dict = self.image_to_dict(self.image_id, image_name, int(img_width), int(img_height))
            self.set_dict_list(image_dict, group, 'images')
            for points_attribute in points_tag_list:
                points_str = points_attribute.attributes['points'].value
                cat = points_attribute.attributes['label'].value

                cat = self.get_category(cat)

                vertices = [list(map(float, x.split(','))) for x in points_str.split(';')]

                for point_idx, vertex in enumerate(vertices):
                    vertex[0] = vertex[0] * resize_factor
                    vertex[1] = vertex[1] * resize_factor
                    if rotate:
                        vertex[0], vertex[1] = vertex[1], img_height - vertex[0]

                vertices = list(itertools.chain(*vertices))


                bb_x = min([vertices[0], vertices[2], vertices[4], vertices[6]])
                bb_y = min([vertices[1], vertices[3], vertices[5], vertices[7]])
                bb_max_x = max([vertices[0], vertices[2], vertices[4], vertices[6]])
                bb_max_y = max([vertices[1], vertices[3], vertices[5], vertices[7]])
                bb = [bb_x, bb_y, bb_max_x - bb_x, bb_max_y - bb_y]
                center = [bb[0] + bb[2] / 2, bb[1] + bb[3] / 2]
                ver = []
                for v_ind, vertex in enumerate(vertices):
                    ver.append(vertex)
                    if (v_ind+1) % 2 == 0:
                        ver.append(2)

                annotation_dict = self.annotation_to_dict(self.annotation_id, self.image_id, cat, bb, img_width,
                                                          img_height, center, ver)
                self.set_dict_list(annotation_dict, group, 'annotations')
                self.set_dict_list(annotation_dict, group, 'annotations')
                self.labels.append(
                    (image_name, center, vertices, bb, cat, rotate, group, self.annotation_id, self.image_id))
                self.inc_annotation_id()
            self.inc_img_id()
                # ver_x = vertices[::2]
                # ver_y = vertices[1::2]
                # ver_x = [x - center[0] for x in ver_x]
                # ver_y = [y - center[1] for y in ver_y]

                # mod = [np.linalg.norm([x,y]) for x,y in zip(ver_x, ver_y)]
                # phase = np.arctan2(ver_y, ver_x )
                # phase = [p if p >= 0 else p+2*np.pi for p in phase]
                # vertices = [[x, ang] for x, ang in zip(mod, phase)]
                # vertices = list(itertools.chain(*vertices))
                # labels.append([vertices, bb, center, cat])

    def draw_annotations(self, img_name, labels):
        resize_factor = self.resize_factor
        img = cv2.imread(self.images_path + img_name)
        thickness = 1
        if resize_factor != 1:
            img = cv2.resize(img, None, interpolation=cv2.INTER_AREA, fx=resize_factor, fy=resize_factor)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        for label in labels:
            vertices, bb, center , cat = label
            color_bb = (250, 0, 0) if cat == 1 else (0, 150, 250)
            color_polygon = (0, 250, 0) if cat == 1 else (0, 150, 190)
            vert_mod = vertices[::2]
            vert_phase = vertices[1::2]
            vertices = [[mod*np.cos(phase), mod*np.sin(phase)] for mod, phase in zip(vert_mod, vert_phase)]
            vertices = list(itertools.chain(*vertices))
            vertices = [int(round(vertex)) for vertex in vertices]
            bb_tl = (int(round(bb[0])), int(round(bb[1])))
            bb_br = (int(round(bb[0] + bb[2])), int(round(bb[1] + bb[3])))
            center = tuple([int(round(center[0])), int(round(center[1]))])
            img = cv2.rectangle(img, bb_tl, bb_br, color_bb, thickness)
            img = cv2.line(img, center, (center[0]+vertices[2], center[1]+vertices[3]), color_polygon, thickness)
            img = cv2.line(img, center, (center[0]+vertices[4], center[1]+vertices[5]), color_polygon, thickness)
            img = cv2.line(img, center, (center[0]+vertices[6], center[1]+vertices[7]), color_polygon, thickness)
            img = cv2.line(img, center, (center[0]+vertices[0], center[1]+vertices[1]), color_polygon, thickness)

            cv2.imwrite(self.annotated_output_dir + img_name, img)

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
            lista = [annotations_dict['image_id'], annotations_dict['center'][0],
                     annotations_dict['center'][1], annotations_dict['vertices'][0],
                     annotations_dict['vertices'][1], annotations_dict['vertices'][3],
                     annotations_dict['vertices'][4], annotations_dict['vertices'][6],
                     annotations_dict['vertices'][7], annotations_dict['vertices'][9],
                     annotations_dict['vertices'][10], annotations_dict['category_id']
                     ]
            annotation_list.append(lista)
        annotations_df = pd.DataFrame(annotation_list, columns=["image_id", 'center_x', 'center_y',
            'v1_x', 'v1_y', 'v2_x', 'v2_y', 'v3_x', 'v3_y', 'v4_x', 'v4_y', 'cat'])
        annotations_ids = annotations_df["image_id"]
        for img_id, image_name in images_dict.items():
            image_path = images_dir + image_name
            img = cv2.imread(image_path)
            ann_df = annotations_df[annotations_df['image_id'] == img_id]
            for idx in range(len(ann_df)):
                image_id = list(ann_df.iloc[idx])[0]
                center_x, center_y = list(ann_df.iloc[idx])[1:3]
                vertices = list(ann_df.iloc[idx])[3:-1]
                cat = list(ann_df.iloc[idx])[-1]
                if cat == '1':
                    color = (23, 220, 75)
                else:
                    color = (220, 23, 75)
                self.draw_polygons(img, vertices, color=color)

                self.draw_points(img, [[center_x, center_y]], radius=5,color=color)
            print(output_dir + image_name)
            cv2.imwrite(output_dir + image_name, img)


class CuadrosTwoCategories(Quadrilateral):
    def get_category(self, cat):
        if cat == "Marco externo":
            return '1'
        else:
            return '2'
    def get_labels_dataframe(self):
        df = []
        for label in self.labels:
            #image_name, center, vertices, bb, cat, rotate, group, self.annotation_id, self.image_id

            df.append([label[0], label[1][0], label[1][1],
                       label[2][0], label[2][1], label[2][2], label[2][3],
                       label[2][4], label[2][5], label[2][6], label[2][7],
                       label[4], label[5], label[6], label[7], label[8]])
        df = pd.DataFrame(df, columns=["image_name", "center_x", "center_y",
                                       "v1_x", "v1_y", "v2_x", "v2_y",
                                       "v3_x", "v3_y", "v4_x", "v4_y",
                                       "cat", "rotate", "group",
                                       "annotation_id", "image_id"])
        df = df.astype({'image_name': str, 'center_x': float, 'center_y': float,
                        "v1_x": float, "v1_y": float, "v2_x": float, "v2_y": float,
                       "v3_x": float, "v3_y": float, "v4_x": float, "v4_y": float,
                        'cat': int, 'rotate': bool, 'group': str, 'annotation_id': int,
                        'image_id': int})
        return df