from grapes import GrapesMixTasksIOU

import os
import json
from xml.dom import minidom



def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__ == "__main__":
    ### cfg ###
    annotations_path_two_frames = "/mnt/datos/capturas/2022-02-17_Zuccardi_Piedra-Infinita_Volumetria_imagenes/F0/annotations.xml" # Archivo qeu contiene el etiquetado
    annotations_path_one_frame = "/mnt/datos/capturas/2022-02-17_Zuccardi_Piedra-Infinita_Volumetria_imagenes/variaciones/annotations.xml" # Archivo qeu contiene el etiquetado
    categories = [{"id": 1, 'name': 'grape', 'supercategory': 'circle'}]

    draw_labels_flag = True
    save_images = True

    images_path_two_frames = '/mnt/datos/capturas/2022-02-17_Zuccardi_Piedra-Infinita_Volumetria_imagenes/F0/'
    images_path_one_frames = '/mnt/datos/capturas/2022-02-17_Zuccardi_Piedra-Infinita_Volumetria_imagenes/variaciones/'
    output_dir = '/mnt/datos/datasets/grapes_mix_iou_reg/'
    annotated_output_dir = output_dir + 'labeled/'
    resize_factor = 0.8  # si vale 1 no modifica el tamaño de las imágenes
    train_percentage = 0.8  # en este caso lo ignoro y uso n_images_train y n_images_test
    n_images_train = 80 # opcional, lo uso si hay muchas imágenes sin etiquetar cuesta calcular la cantidad de elemetos el conjunto de train y test
    n_images_test = 15 # opcional, lo uso si hay muchas imágenes sin etiquetar cuesta calcular la cantidad de elemetos el conjunto de train y test
    dataset = 'grapes_two_frames'
    rm_two_frames = ['VID_20220217_111141_F0.png',
          'VID_20220217_103734_F0.png',
          # 'VID_20220217_102337_F0.png',
          # 'VID_20220217_105114_F0.png',
          'VID_20220217_105534_F0.png',
          ] # cómo hay imágenes que por alguna razón interrumpen el flujo ejecución, las elimino del análisis
    prefix_two_Frames = ['optima/cvat/2022-02-17_Zuccardi_Piedra-Infinita_Volumetria/merge2/', '']
    suffix_two_frames = ['merge', 'F0']
    dict_list = {"info": {"description": "grapes", "url": "", "version": "0.0.1", "year": 2022, "contributor": "", "date_created": ""}, "licenses": [{"id": 1, "name": "Attribution-NonCommercial-ShareAlike License", "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"}], "categories": categories, "images": [], "annotations": []}

    def image_to_dict(image_id, file_name, width, height):
        return {"id": image_id, "file_name": file_name, "width": width, "height": height, "date_captured": "",
                "license": 1, "coco_url": "", "flickr_url": ""}

    def annotation_to_dict(ann_id, im_id, cat, bb, width, height, center, radius, occlusion_factor, points):
        return {"id": ann_id, "image_id": im_id, "category_id": cat, "iscrowd": 0, "area": 0,
                "bbox": [bb[0], bb[1], bb[2], bb[3]], "width": width, "height": height,
                "circle_center": center, 'circle_radius': radius, 'occlusion_factor': occlusion_factor, "points": points}

    ############


    xml_doc_two_frames = minidom.parse(annotations_path_two_frames)

    coco_parser_two_frames = GrapesMixTasksIOU(
        images_path_two_frames,
        xml_doc_two_frames,
        train_percentage,
        resize_factor,
        dict_list,
        image_to_dict,
        annotation_to_dict
    )
    coco_parser_two_frames.set_two_frames()
    coco_parser_two_frames.set_prefix(prefix_two_Frames)
    coco_parser_two_frames.set_suffix(suffix_two_frames)
    coco_parser_two_frames.set_rm(rm_two_frames)
    coco_parser_two_frames.set_n_images_train(n_images_train)
    coco_parser_two_frames.parse_to_coco()



    xml_doc_one_frame = minidom.parse(annotations_path_one_frame)
    coco_parser_one_frame = GrapesMixTasksIOU(
        images_path_one_frames,
        xml_doc_one_frame,
        train_percentage,
        resize_factor,
        dict_list,
        image_to_dict,
        annotation_to_dict
    )
    coco_parser_one_frame.image_id = coco_parser_two_frames.image_id+1
    coco_parser_one_frame.annotation_id = coco_parser_two_frames.annotation_id+1
    coco_parser_one_frame.dict_list_test = coco_parser_two_frames.dict_list_test
    coco_parser_one_frame.dict_list_train = coco_parser_two_frames.dict_list_train
    coco_parser_one_frame.labels = coco_parser_two_frames.labels
    coco_parser_one_frame.set_n_images_train(80+coco_parser_two_frames.image_id)
    coco_parser_one_frame.parse_to_coco()

    make_dir(output_dir)
    json.dump(coco_parser_one_frame.get_train_dict(), open(output_dir+'train.json', 'w'))
    json.dump(coco_parser_one_frame.get_test_dict(), open(output_dir+'test.json', 'w'))

    dir_train = output_dir + 'train/'
    dir_test = output_dir + 'test/'
    if save_images:

        make_dir(dir_train)
        make_dir(dir_test)
        dir_source = images_path_one_frames
        coco_parser_one_frame.save_images(dir_source, dir_train, 'train')
        coco_parser_one_frame.save_images(dir_source, dir_test, 'test')
        dir_source = images_path_two_frames
        coco_parser_one_frame.save_images(dir_source, dir_train, 'train')
        coco_parser_one_frame.save_images(dir_source, dir_test, 'test')

    if draw_labels_flag:
        make_dir(annotated_output_dir)
        coco_parser_one_frame.draw_labels(output_dir+'test.json', dir_test, annotated_output_dir)
        coco_parser_one_frame.draw_labels(output_dir+'train.json', dir_train, annotated_output_dir)

