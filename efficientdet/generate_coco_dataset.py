"""
Script to generate a COCO style dataset from the
Global Wheat Challenge dataset structure.

This file structure can be used to train a model using
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch 
"""
import ast
import json
import os
import shutil
import yaml

from multiprocessing import Pool
from tqdm import tqdm
from ruamel.yaml.scalarstring import SingleQuotedScalarString as sq

def generate_annotations(df):
    """
    Expects a dataframe with columns:
    image_id
    width
    height
    bbox: [x, y, w, h]
    """
    annot = {
        "info": {
            "description": "",
            "url": "",
            "version": "",
            "year": 2020,
            "contributor": "",
            "date_created": ""
        },
        "licenses": [{
            "id": 1,
            "name": None,
            "url": None
        }],
        "categories": [{
            "id": 1,
            "name": "wheat",
            "supercategory": "None"
        }],
        "images" : [],
        "annotations": []
    }

    annot_id = 0

    for i, img in tqdm(enumerate(df['image_id'].unique())):
        dimg = df[df['image_id'] == img]
        image_id = i+1
        im = {
            "id": image_id,
            "file_name": f"{img}.jpg",
            "width": int(df['width'].iloc[0]),
            "height": int(df['height'].iloc[0]),
            "date_captured": "",
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        }

        annot['images'].append(im)

        for (box, (w, h)) in zip(dimg['bbox'].values, dimg[['w', 'h']].values):
            annot_id += 1

            box = ast.literal_eval(box)

            ann = {
                "id": annot_id,
                "image_id": image_id,
                "category_id": 1,
                "iscrowd": 0,
                "area": float(w * h),
                "bbox": box,
                "segmentation": []
            }

            annot['annotations'].append(ann)

    return annot


def generate_params(project_name):
    """
    """
    params = {
        "project_name": project_name,
        "train_set": "train",
        "val_set": "val",
        "num_gpus": 1,

        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],

        "anchors_scales": "[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]",
        "anchors_ratios": "[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]",

        "obj_list": ["wheat"]
    }

    return params

def generate_coco_dataset(base_dir, df_train, df_val, target_base_dir, project_name):
    """
    Generate COCO dataset from the wheat dataset directory structure.
    """
    data_dir = os.path.join(target_base_dir, f'datasets/{project_name}')

    # move images from base_dir/data/train to target directory
    os.makedirs(os.path.join(data_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val'), exist_ok=True)

    # def cptrain(imid):
    #     return shutil.copyfile(os.path.join(base_dir, f'data/train/{imid}.jpg'),
    #                            os.path.join(data_dir, f'train/{imid}.jpg'))
    
    # with Pool() as p:
    #     tqdm(p.imap_unordered(cptrain, df_train['image_id'].unique()))

    for imid in tqdm(df_train['image_id'].unique()):
        shutil.copyfile(os.path.join(base_dir, f'data/train/{imid}.jpg'),
                        os.path.join(data_dir, f'train/{imid}.jpg'))
    
    # def cpval(imid):
    #     return shutil.copyfile(os.path.join(base_dir, f'data/train/{imid}.jpg'),
    #                            os.path.join(data_dir, f'val/{imid}.jpg'))


    # with Pool() as p:
    #     tqdm(p.imap_unordered(cpval, df_val['image_id'].unique()))

    for imid in tqdm(df_val['image_id'].unique()):
        shutil.copyfile(os.path.join(base_dir, f'data/train/{imid}.jpg'),
                        os.path.join(data_dir, f'val/{imid}.jpg'))

    # create annotations
    os.makedirs(os.path.join(data_dir, 'annotations'), exist_ok=True)

    annot_train = generate_annotations(df_train)
    annot_val = generate_annotations(df_val)

    with open(os.path.join(data_dir, 
                           'annotations', 
                           'instances_train.json'), 'w') as f:
        json.dump(annot_train, f)

    with open(os.path.join(data_dir, 
                           'annotations', 
                           'instances_val.json'), 'w') as f:
        json.dump(annot_val, f)

    # create parameters file
    params = generate_params(project_name)

    with open(os.path.join(target_base_dir,
                           f'projects/{project_name}.yml'), 'w') as f:
        yaml.dump(params, f)
