from pycocotools.coco import COCO

coco_filename = "coco_example.json"

dlib_save_filename = "output.txt"
coco_dataset = COCO(coco_filename)

image_ids = coco_dataset.getImgIds()
# category_ids = coco_dataset.getCatIds()
# annotations = coco_dataset.loadAnns(coco_dataset.getAnnIds(imgIds=[image_ids[0]]))
category_ids = coco_dataset.getCatIds(catNms=['person'])

with open(dlib_save_filename, 'w') as f:

    print("# Data Directory:\n", file=f)
    print("# file location, {x,y,w,h,label}, {x,y,w,h,label}, ...", file=f)

    for idx in range(len(image_ids)):
        # first_image_id = image_ids[0]
        annotation_ids = coco_dataset.getAnnIds(imgIds=image_ids[idx], catIds=category_ids, iscrowd=None)
        annotations = coco_dataset.loadAnns(annotation_ids)

        if len(annotations) > 0:
            # print('{},'.format(coco_dataset.imgs[image_ids[idx]]['file_name']))
            s_line = coco_dataset.imgs[image_ids[idx]]['file_name'] + ","

            for jdx in range(len(annotations)):
                bbox = ",".join(map(str,annotations[jdx]['bbox']))
                s_line += "{" + bbox + ",person},"

            # print("{}".format(s_line[:-1]))
            print("{}".format(s_line[:-1]), file=f)

bp = 1
