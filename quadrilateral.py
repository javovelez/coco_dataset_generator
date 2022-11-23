from coco_manager import *
import numpy as np
class Quadrilateral(CocoManager):

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


    def parse_to_coco(self):
        resize_factor = self.resize_factor
        xml_image_list = self.xml_images_list
        n_images_total = xml_image_list.length
        n_images = int(round(n_images_total * self.images_perc))

        for image in xml_image_list:
            img_xml_id = image.attributes["id"].value
            if not (start <= int(img_xml_id) < start + n_images):
                continue
            img_xml_name = image.attributes["name"].value
            self.add_image(img_xml_name)
            points_tag_list = image.getElementsByTagName('points')

            labels = []

            for points_attribute in points_tag_list:
                points_str = points_attribute.attributes['points'].value
                cat = points_attribute.attributes['label'].value

                if cat == 'Marco interno' or cat == 'cara frontal heladera':
                    cat = 1
                else:
                    cat = 2

                vertices = [list(map(float, x.split(','))) for x in points_str.split(';')]

                vertices = np.asarray(vertices) * resize_factor
                vertices = list(itertools.chain(*vertices))
                bb_x = min([vertices[0], vertices[2], vertices[4], vertices[6]])
                bb_y = min([vertices[1], vertices[3], vertices[5], vertices[7]])
                bb_max_x = max([vertices[0], vertices[2], vertices[4], vertices[6]])
                bb_max_y = max([vertices[1], vertices[3], vertices[5], vertices[7]])
                bb = [bb_x, bb_y, bb_max_x - bb_x, bb_max_y - bb_y]
                center = [bb[0] + bb[2] / 2, bb[1] + bb[3] / 2]

                ver_x = vertices[::2]
                ver_y = vertices[1::2]
                ver_x = [x - center[0] for x in ver_x]
                ver_y = [y - center[1] for y in ver_y]

                mod = [np.linalg.norm([x,y]) for x,y in zip(ver_x, ver_y)]
                phase = np.arctan2(ver_y, ver_x )
                phase = [p if p >= 0 else p+2*np.pi for p in phase]
                vertices = [[x, ang] for x, ang in zip(mod, phase)]
                vertices = list(itertools.chain(*vertices))
                labels.append([vertices, bb, center, cat])