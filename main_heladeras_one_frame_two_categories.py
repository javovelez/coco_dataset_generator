from grapes import GrapesTwoFramesCVATtasks159Job144
from quadrilateral import Quadrilateral

import os
import json
from xml.dom import minidom

coco_dataset_factory = {
    'grapes_two_frames': GrapesTwoFramesCVATtasks159Job144,
    'quadrilateral': Quadrilateral,
}

def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__ == "__main__":
    ### cfg ###
    annotations_path = "./annotations.xml" # Archivo qeu contiene el etiquetado
    categories = [{"id": 1, 'name': 'front', 'supercategory': 'quadrilateral'}, {"id": 1, 'name': 'lateral', 'supercategory': 'quadrilateral'}]

    draw_labels_flag = True
    save_images = True

    images_path = '/mnt/datos/capturas/2022-02-17_Zuccardi_Piedra-Infinita_Volumetria_imagenes/F0/'
    output_dir = '/mnt/datos/datasets/dataset_fridges/'
    annotated_output_dir = output_dir + 'labeled/'
    resize_factor = 1  # si vale 1 no modifica el tamaño de las imágenes
    train_percentage = 0.8  # en este caso lo ignoro y uso n_images_train y n_images_test
    # n_images_train = 80 # opcional, lo uso si hay muchas imágenes sin etiquetar cuesta calcular la cantidad de elemetos el conjunto de train y test
    # n_images_test = 15 # opcional, lo uso si hay muchas imágenes sin etiquetar cuesta calcular la cantidad de elemetos el conjunto de train y test
    dataset = 'grapes_two_frames'
    rm = [] # cómo hay imágenes que por alguna razón interrumpen el flujo ejecución, las elimino del análisis
    prefix = ['optima/cvat/2022-02-17_Zuccardi_Piedra-Infinita_Volumetria/merge2/', '']
    suffix = ['merge', 'F0']
    dict_list = {"info": {"description": "fridges", "url": "", "version": "0.0.1", "year": 2022, "contributor": "", "date_created": ""}, "licenses": [{"id": 1, "name": "Attribution-NonCommercial-ShareAlike License", "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"}], "categories": categories, "images": [], "annotations": []}

    def image_to_dict(image_id, file_name, width, height):
        return {"id": image_id, "file_name": file_name, "width": width, "height": height, "date_captured": "",
                "license": 1, "coco_url": "", "flickr_url": ""}

    def annotation_to_dict(ann_id, im_id, cat, bb, width, height, center, radius):
        return {"id": ann_id, "image_id": im_id, "category_id": cat, "iscrowd": 0, "area": 0,
                "bbox": [bb[0], bb[1], bb[2], bb[3]], "width": width, "height": height,
                "center": center, 'radius': radius}

    ############


    xml_doc = minidom.parse(annotations_path)

    coco_parser = coco_dataset_factory[dataset](
        'train',
        images_path,
        xml_doc,
        train_percentage,
        resize_factor,
        dict_list,
        image_to_dict,
        annotation_to_dict
    )
    coco_parser.set_prefix(prefix)
    coco_parser.set_suffix(suffix)
    coco_parser.set_rm(rm)
    # coco_parser.set_n_images_train(n_images_train)
    # coco_parser.set_n_images_test(n_images_test)
    coco_parser.parse_to_coco()
    make_dir(output_dir)
    json.dump(coco_parser.get_train_dict(), open(output_dir+'train.json', 'w'))
    json.dump(coco_parser.get_test_dict(), open(output_dir+'test.json', 'w'))

    dir_train = output_dir + 'train/'
    dir_test = output_dir + 'test/'
    if save_images:

        make_dir(dir_train)
        make_dir(dir_test)
        coco_parser.save_images(dir_train, 'train')
        coco_parser.save_images(dir_test, 'test')

    if draw_labels_flag:
        make_dir(annotated_output_dir)
        coco_parser.draw_labels(output_dir+'test.json', dir_test, annotated_output_dir)
        coco_parser.draw_labels(output_dir+'train.json', dir_train, annotated_output_dir)

