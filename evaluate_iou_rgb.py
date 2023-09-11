#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import sys
import numpy as np
import torch
from PIL import Image
from collections import defaultdict

from modules.ioueval import iouEval
from common.laserscan import SemLaserScan

# possible splits
splits = ['train','valid','test']
def save_to_log(logdir,logfile,message):
    f = open(logdir+'/'+logfile, "a")
    f.write(message+'\n')
    f.close()
    return

def eval(test_sequences,splits,pred):
    # get scan paths
    scan_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        scan_paths = os.path.join(FLAGS.dataset, "sequences",
                                  str(sequence), "velodyne")
        # populate the scan names
        seq_scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(scan_paths)) for f in fn if ".bin" in f]
        seq_scan_names.sort()
        scan_names.extend(seq_scan_names)
    # print(scan_names)

    # get label paths
    label_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        label_paths = os.path.join(FLAGS.dataset, "sequences",
                                   str(sequence), "labels")
        # populate the label names
        seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(label_paths)) for f in fn if ".label" in f]
        seq_label_names.sort()
        label_names.extend(seq_label_names)
    # print(label_names)

    # get predictions paths
    pred_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        pred_paths = os.path.join(FLAGS.predictions, "sequences",
                                  sequence, "predictions")
        # populate the label names
        seq_pred_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(pred_paths)) for f in fn if ".label" in f]
        seq_pred_names.sort()
        pred_names.extend(seq_pred_names)
    # print(pred_names)

    rgb_files = []
    # fill in with names, checking that all sequences are complete
    for sequence in test_sequences:
      # to string
      sequence = '{0:02d}'.format(int(sequence))
      rgb_paths = os.path.join(FLAGS.dataset, "sequences", sequence, "image_2")

      # get files
      seq_rgb_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(rgb_paths)) for f in fn if ".png" in f]

      # extend list
      rgb_files.extend(seq_rgb_files)

#     tt = 1000
#     scan_names = scan_names[0:tt]
#     label_names = label_names[0:tt]
#     pred_names = pred_names[0:tt]

    # check that I have the same number of files
    # print("labels: ", len(label_names))
    # print("predictions: ", len(pred_names))
    assert (len(label_names) == len(scan_names) and
            len(label_names) == len(pred_names) and
            len(label_names) == len(rgb_files))

    aug_prob = defaultdict(lambda: -1.0)
    aug_prob["point_dropping"] = [-1.0, -1.0]

    print("Evaluating sequences: ")
    # open each file, get the tensor, and make the iou comparison
    for scan_file, label_file, pred_file, rgb_file in zip(scan_names, label_names, pred_names, rgb_files):
        print("evaluating label ", label_file, "with", pred_file)
        # open label
        rgb_data = Image.open(rgb_file)
        label = SemLaserScan(project=False, aug_prob=aug_prob, only_rgb=True)
        label.open_scan(scan_file, rgb_data)
        label.open_label(label_file)
        u_label_sem = remap_lut[label.sem_label]  # remap to xentropy format
        if FLAGS.limit is not None:
            u_label_sem = u_label_sem[:FLAGS.limit]

        # open prediction
        pred = SemLaserScan(project=False, aug_prob=aug_prob, only_rgb=True)
        pred.open_scan(scan_file, rgb_data)
        pred.open_label(pred_file)
        u_pred_sem = remap_lut[pred.sem_label]  # remap to xentropy format
        if FLAGS.limit is not None:
            u_pred_sem = u_pred_sem[:FLAGS.limit]

        # add single scan to evaluation
        evaluator.addBatch(u_pred_sem, u_label_sem)
        front_evaluator.addBatch(u_pred_sem, u_label_sem)

    # when I am done, print the evaluation
    m_accuracy = evaluator.getacc()
    m_jaccard, class_jaccard = evaluator.getIoU()
    m_accuracy_front = front_evaluator.getacc()
    m_jaccard_front, class_jaccard_front = front_evaluator.getIoU()

    print('{split} set:\n'
          'Acc avg {m_accuracy:.3f}\n'
          'IoU avg {m_jaccard:.3f}\n'
          'Acc front avg {m_accuracy_front:.3f}\n'
          'IoU front avg {m_jaccard_front:.3f}'.format(split=splits,
                                           m_accuracy=m_accuracy,
                                           m_jaccard=m_jaccard,
                                           m_accuracy_front=m_accuracy_front,
                                           m_jaccard_front=m_jaccard_front))

    save_to_log(FLAGS.predictions,'pred.txt','{split} set:\n'
          'Acc avg {m_accuracy:.3f}\n'
          'IoU avg {m_jaccard:.3f}\n'
          'Acc front avg {m_accuracy_front:.3f}\n'
          'IoU front avg {m_jaccard_front:.3f}'.format(split=splits,
                                           m_accuracy=m_accuracy,
                                           m_jaccard=m_jaccard,
                                           m_accuracy_front=m_accuracy_front,
                                           m_jaccard_front=m_jaccard_front))
    # print also classwise
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc))
            save_to_log(FLAGS.predictions, 'pred.txt', 'IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc))
            
    for i, jacc in enumerate(class_jaccard_front):
        if i not in ignore:
            print('IoU front class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc))
            save_to_log(FLAGS.predictions, 'pred.txt', 'IoU front class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc))

    # print for spreadsheet
    print("*" * 80)
    print("IoU below can be copied straight for paper table")
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            sys.stdout.write('{jacc:.3f}'.format(jacc=jacc.item()))
            sys.stdout.write(",")
    sys.stdout.write('{jacc:.3f}'.format(jacc=m_jaccard.item()))
    sys.stdout.write(",")
    sys.stdout.write('{acc:.3f}'.format(acc=m_accuracy.item()))
    sys.stdout.write('\n')
    sys.stdout.flush()

    print("*" * 80)
    print("IoU front below can be copied straight for paper table")
    for i, jacc in enumerate(class_jaccard_front):
        if i not in ignore:
            sys.stdout.write('{jacc:.3f}'.format(jacc=jacc.item()))
            sys.stdout.write(",")
    sys.stdout.write('{jacc:.3f}'.format(jacc=m_jaccard_front.item()))
    sys.stdout.write(",")
    sys.stdout.write('{acc:.3f}'.format(acc=m_accuracy_front.item()))
    sys.stdout.write('\n')
    sys.stdout.flush()

    print(evaluator.conf_matrix)
    print(front_evaluator.conf_matrix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./evaluate_iou.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=False,
        default="../SemanticKITTI/dataset",
        help='Dataset dir. No Default',
    )
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        required=False,
        default="logs/fusion_without",
        help='Prediction dir. Same organization as dataset, but predictions in'
             'each sequences "prediction" directory. No Default. If no option is set'
             ' we look for the labels in the same directory as dataset'
    )
    parser.add_argument(
        '--split', '-s',
        type=str,
        required=False,
        choices=["train", "valid", "test"],
        default="valid",
        help='Split to evaluate on. One of ' +
             str(splits) + '. Defaults to %(default)s',
    )
    parser.add_argument(
        '--data_cfg', '-dc',
        type=str,
        required=False,
        default="config/labels/semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        required=False,
        default=None,
        help='Limit to the first "--limit" points of each scan. Useful for'
             ' evaluating single scan from aggregated pointcloud.'
             ' Defaults to %(default)s',
    )

    FLAGS, unparsed = parser.parse_known_args()

    # fill in real predictions dir
    if FLAGS.predictions is None:
        FLAGS.predictions = FLAGS.dataset

    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print("Data: ", FLAGS.dataset)
    print("Predictions: ", FLAGS.predictions)
    print("Split: ", FLAGS.split)
    print("Config: ", FLAGS.data_cfg)
    print("Limit: ", FLAGS.limit)
    print("*" * 80)

    # assert split
    assert (FLAGS.split in splits)

    # open data config file
    try:
        print("Opening data config file %s" % FLAGS.data_cfg)
        DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # get number of interest classes, and the label mappings
    class_strings = DATA["labels"]
    class_remap = DATA["learning_map"]
    class_inv_remap = DATA["learning_map_inv"]
    class_ignore = DATA["learning_ignore"]
    nr_classes = len(class_inv_remap)

    # make lookup table for mapping
    maxkey = 0
    for key, data in class_remap.items():
        if key > maxkey:
            maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in class_remap.items():
        try:
            remap_lut[key] = data
        except IndexError:
            print("Wrong key ", key)
    # print(remap_lut)

    # create evaluator
    ignore = []
    for cl, ign in class_ignore.items():
        if ign:
            x_cl = int(cl)
            ignore.append(x_cl)
            print("Ignoring xentropy class ", x_cl, " in IoU evaluation")

    # create evaluator
    device = torch.device("cpu")
    evaluator = iouEval(nr_classes, device, ignore)
    front_evaluator = iouEval(nr_classes, device, ignore)
    evaluator.reset()
    front_evaluator.reset()

    # get test set
    if FLAGS.split is None:
        for splits in ('train','valid'):
            eval((DATA["split"][splits]),splits,FLAGS.predictions)
    else:
        eval(DATA["split"][FLAGS.split],splits,FLAGS.predictions)