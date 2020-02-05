import numpy as np
import PIL.Image
import argparse
import glob
import os
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True)
parser.add_argument('--output_dir', required=True)
args = parser.parse_args()


# To evaluate Cityscapes results on the evaluation server, the labels used
# during training should be mapped to the labels for evaluation.
_CITYSCAPES_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                   23, 24, 25, 26, 27, 28, 31, 32, 33]
_CITYSCAPES_EVAL_ID_TO_TRAIN_ID = {n: i for i,n in enumerate(_CITYSCAPES_TRAIN_ID_TO_EVAL_ID)}
print(_CITYSCAPES_EVAL_ID_TO_TRAIN_ID)

def _convert_eval_id_to_train_id(prediction, eval_id_to_train_id=_CITYSCAPES_EVAL_ID_TO_TRAIN_ID):
    """Converts the predicted label for evaluation.

    There are cases where the training labels are not equal to the evaluation
    labels. This function is used to perform the conversion so that we could
    evaluate the results on the evaluation server.

    Args:
      prediction: Semantic segmentation prediction.
      train_id_to_eval_id: A list mapping from train id to evaluation id.

    Returns:
      Semantic segmentation prediction whose labels have been changed.
    """
    converted_prediction = prediction.copy()
    for eval_id, train_id in eval_id_to_train_id.items():
        converted_prediction[prediction == eval_id] = train_id

    return converted_prediction

if __name__ == '__main__':

    input_files = glob.glob(os.path.join(args.input_dir, "*"))
    assert os.path.exists(args.input_dir)
    assert len(input_files) > 0

    if not os.path.exists(args.output_dir):
        print('Creating output dir at: {}'.format(args.output_dir))
        os.makedirs(args.output_dir)

    for input_file in tqdm.tqdm(input_files):
        output_file = os.path.join(args.output_dir,
                                    os.path.basename(input_file))
        img = np.array(PIL.Image.open(input_file))
        conv = _convert_eval_id_to_train_id(img)
        PIL.Image.fromarray(conv).save(output_file)
