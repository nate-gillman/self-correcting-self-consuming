import sys

# system path is '/oscar/data/superlab/users/nates_stuff/motion-diffusion-model/scripts', want to go down one
for pth in sys.path:
    if "motion-diffusion-model/scripts" in pth:
        sys.path.append(pth[:-8])
        break

from os.path import join as pjoin
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from sample.generate import loop_generate
from utils.parser_util import generate_args




def get_motions_idxs_and_captions_with_keyword(keyword, data_root):
    # ./dataset/HumanML3D/train.txt
    split = "train"
    split_file = pjoin(data_root, f"{split}.txt")

    # ./dataset/HumanML3D/texts/
    text_dir = pjoin(data_root, "texts")

    id_list = []

    with open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())

    # new_name_list: List[str] = []
    # length_list: List[int] = []

    def _get_codec_lines(text_dir: str, name: str) -> List[str]:
        rl = []
        with open(pjoin(text_dir, name + ".txt")) as f:
            rl = f.readlines()
        return rl

    def _parse_datapoint_line(line: str) -> Tuple[str, List[str], float, float]:
        line_split = line.strip().split("#")
        # ex. "a man kicks something or someone with his left leg"
        caption = line_split[0]
        # ex. "a/DET man/NOUN kick/VERB something/PRON or/CCONJ someone/PRON with/ADP his/DET left/ADJ leg/NOUN"

        tokens = line_split[1].split(" ")
        # text_dict example:
        # {
        #    "caption": "a man kicks something or someone with his left leg"
        #    "tokens": ["a/DET", "man/NOUN", "kick/VERB", "something/PRON"...]
        # }
        # --- extract motion sequence part ---
        # in a video, the motion may be too complicated to describe in one caption.
        # several parts of a motion are broken up and the caption may describe a
        # section of the video. The f_tag and to_tag define these segments of a video
        # in seconds. They are 0.0 by default.
        # ex. "0.0", start tag
        f_tag = float(line_split[2])

        # ex. "0.0", end tag
        to_tag = float(line_split[3])

        # default to 0.0 if no interval for caption is given
        f_tag = 0.0 if np.isnan(f_tag) else f_tag
        to_tag = 0.0 if np.isnan(to_tag) else to_tag

        return caption, tokens, f_tag, to_tag

    # --- iterate through dataset loaded from disk ---
    idx_to_caption = {}
    print(
        f"Iterating through dataset, searching for motion clips containing '{keyword}'"
    )
    for name in tqdm(id_list):
        try:
            # process the text data associated with the motion sequence
            lines = map(
                _parse_datapoint_line,
                _get_codec_lines(text_dir, name),
            )
            for line in lines:
                caption, tokens, _, _ = line
                if keyword in caption and name not in idx_to_caption:
                    # skip this sample if it does NOT contain the given keyword,
                    # OR if it's already been saved
                    idx_to_caption[name] = caption
        except:
            pass

    # idxs = sorted(list(set(idxs))) # remove duplicates

    return idx_to_caption


def get_prompts_from(idx_to_caption):
    prompts = []
    for idx, caption in idx_to_caption.items():
        motion_dict = {}
        motion_dict["text"] = caption
        motion_dict["file"] = idx + "-sampled.npy"
        prompts.append(motion_dict)

    return prompts


if __name__ == "__main__":
    """
    Example generation command from list of text prompts:
        python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt --input_text ./assets/example_text_prompts.txt --num_repetitions 1


    Example generation to save individual .npy files for batched generated motion for a dictionary of
    text prompts:
        python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt --output_dir ./save/batch_generated_multiprompt --is_generate_batched_text_prompt_samples True

    """
    args = generate_args()

    keyword = ""  # i.e., filter by trivial keyword, so include everything...
    idx_to_caption = get_motions_idxs_and_captions_with_keyword(keyword)
    prompts = get_prompts_from(idx_to_caption)

    assert (
        args.is_generate_batched_text_prompt_samples
    ), "args.is_generate_batched_text_prompt_samples"

    # default batch size is 64
    # 64 * 10 prompts, with 1,000 diffusion steps takes ~1,233 seconds (incl. dataset / model checkpoint load time)
    # approx: 40 sec + 1.86 sec/prompt

    loop_generate(prompts)
