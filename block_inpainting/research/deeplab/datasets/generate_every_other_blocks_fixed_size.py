import os
import PIL.Image
import numpy as np
import fnmatch
import tqdm

'''
Pseudo-checkerboard block sampling. PERC_LABELLED dictates number of blocks to sample.
'''

B = 10
ignore_label = 255

PERC_LABELLED = 0.5
assert 0 < PERC_LABELLED <= 0.5
input_label_dir = "cityscapes/gtFine"
output_label_dir = "cityscapes/gtFine_every_other_block_B{}p{}".format(B, PERC_LABELLED)
vis_output_label_dir = "cityscapes/gtFine_every_other_block_B{}p{}_vis".format(B, PERC_LABELLED)

input_files = []
for root, dirnames, filenames in os.walk(input_label_dir):
    for filename in fnmatch.filter(filenames, "*Train*.png"):
        input_files.append(os.path.join(root, filename))
input_files = sorted([f for f in input_files if 'test' not in f])

vis_input_files = []
for root, dirnames, filenames in os.walk(input_label_dir):
    for filename in fnmatch.filter(filenames, "*color.png"):
        vis_input_files.append(os.path.join(root, filename))
vis_input_files = sorted([f for f in vis_input_files if 'test' not in f])

assert len(input_files) == len(vis_input_files), (len(input_files), len(vis_input_files))

ignore_avg = []
for file, vis_file in tqdm.tqdm(zip(input_files, vis_input_files), total=len(input_files)):
    label_arr = np.array(PIL.Image.open(file).convert("L"), dtype=np.uint8)
    vis_label_arr = np.array(PIL.Image.open(vis_file).convert("RGB"), dtype=np.uint8)

    block_x_step = int(np.ceil(label_arr.shape[0] / float(B)))
    block_y_step = int(np.ceil(label_arr.shape[1] / float(B)))


    num_skip = int(1.0 / PERC_LABELLED)

    # Shift 50% of time
    if np.random.random() > 10.5:
        skip = num_skip // 4 - 1
    else:
        skip = 0

    labelled = 0
    for i in range(0, label_arr.shape[0]  - 1, block_x_step):
        skip += 1
        for j in range(0, label_arr.shape[1]  - 1, block_y_step):
            if skip % num_skip != 0: # Set to zero.
                skip += 1
                i_end = i + block_x_step
                j_end = j + block_y_step
                label_arr[i:i_end, j:j_end] = ignore_label
                vis_label_arr[i:i_end, j:j_end] = 0
            else:
                skip = 1
                labelled += 1

    savename = file.replace('_gtFine_labelTrainIds.png', '_B{}PERC{}.png'.format(B, PERC_LABELLED))
    savename = savename.replace(input_label_dir, output_label_dir)
    if not os.path.exists(os.path.dirname(savename)):
        os.makedirs(os.path.dirname(savename))
    PIL.Image.fromarray(label_arr).save(savename)

    vis_savename = vis_file.replace('_gtFine_color.png', '_B{}PERC{}.png'.format(B, PERC_LABELLED))
    vis_savename = vis_savename.replace(input_label_dir, vis_output_label_dir)
    if not os.path.exists(os.path.dirname(vis_savename)):
        os.makedirs(os.path.dirname(vis_savename))
    PIL.Image.fromarray(vis_label_arr).save(vis_savename)

    ignore_avg.append(float(np.sum(label_arr == ignore_label)) / np.size(label_arr))

print("AVG IGNORE LABEL PER IMAGE: {}".format(np.mean(ignore_avg)))


