
# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
from cgi import print_arguments

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import numpy
import os

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', default='/home/seungjae/lfin/example/', help='Image file')
    parser.add_argument('--config', default='./configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco.py', help='Config file')
    parser.add_argument('--checkpoint', default='./checkpoints/mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    
    # path = '/home/seungjae/lfin/example/detection/raw/'
    path = '/media/hard/LfinDataset_raw/images/'
    file_list = sorted(os.listdir(path), reverse=True)

    for files in file_list:
        for file in sorted(os.listdir(path + files)):
            
            if int(files) > 45000 and int(files) < 51121 and not "mp4" in file:

                print("0. File:", file)

                abs_file = path + files + "/" + file

                result = inference_detector(model, abs_file)
                
                if isinstance(result, tuple):
                    bbox_result, segm_result = result
                    if isinstance(segm_result, tuple):
                        segm_result = segm_result[0]  # ms rcnn
                else:
                    bbox_result, segm_result = result, None
                bboxes = numpy.vstack(bbox_result)
                labels = [
                    numpy.full(bbox.shape[0], i, dtype=numpy.int32)
                    for i, bbox in enumerate(bbox_result)
                ]
                labels = numpy.concatenate(labels)

                assert bboxes is not None and bboxes.shape[1] == 5
                scores = bboxes[:, -1]
                inds = scores > 0.3
                bboxes = bboxes[inds, :]
                labels = labels[inds]

                # save_flag = False
                # dyn_obj = [0, 1, 2, 3, 5]
                # for o in dyn_obj:
                #     if o in labels:
                #         print("1. Find: ", o, "in", labels, "Save Image")
                #         save_flag = True
                #         break
                
                # if save_flag:
                    # white masking
                filename = file.split('.')[0]
                mask_img_file = '/home/sj98lee/masked_images/' + files + "/" + filename + '_masked.jpg'
                print(mask_img_file)
                model.show_result(abs_file, result, font_size=0, thickness=0, mask_color=(0,0,0), out_file=mask_img_file)


            # print("== Result ==")

            # labeled masking
            
            # model.show_result(path+file, result, font_size=50, out_file=path+'../labeled/'+name[0]+'_result.jpg')

            # print("\n=== Result ===\n" , result)

           
            # show the results
            # show_result_pyplot(
            #     model,
            #     img_file,
            #     result,
            #     palette=args.palette,
            #     score_thr=args.score_thr)

            # input("Press enter to exit...")


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
