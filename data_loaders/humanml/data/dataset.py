import codecs as cs
import hashlib
import os
import random
from argparse import Namespace
from os.path import join as pjoin
from typing import Dict, List, Optional, Tuple

import numpy as np
import spacy  # type: ignore
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from data_loaders.humanml.common.mean_variance import get_mean_std
from data_loaders.humanml.utils.get_opt import get_opt
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from sample.types import (
    T_BATCH,
    T_DATA_DICT,
    T_HUMANML3D_KIT_DATASET_MODE,
    T_HUMANML3D_KIT_DATASET_SPLIT_TYPE,
    T_IDX_COLLATED_DIFFUSION_SAMPLE,
    GeneratedSampleBatchDict,
    MotionSequenceSampleDict,
    MotionTextSampleDict,
)

_T_GET_ITEM = Tuple[T_BATCH, int]


def collate_fn(batch) -> torch.Tensor:
    """
    See: data_loaders/tensors.py
    """
    # batch elements are (batch, idx), remove the idx tuple
    batch_ext = [x[0] for x in batch]
    batch_ext.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch_ext)


def load_pickled_np_dict(from_path: str) -> Dict:
    saved_data = np.load(from_path, allow_pickle=True)

    # need to do this strange indexing when we save a python dictionary in .npy file
    # we will recover the original object this way
    return saved_data[None][0]


def inverse_transform(data, mean, std) -> torch.Tensor:
    return data * std + mean


class BaseDatasetMDM(data.Dataset):
    def __init__(self, std, mean) -> None:
        super().__init__()
        self.std = std
        self.mean = mean

    def inv_transform(self, data: torch.Tensor) -> torch.Tensor:
        return inverse_transform(data, self.mean, self.std)


class Text2MotionDataset(data.Dataset):
    """For use of training text-2-motion generative model"""

    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        min_motion_len = 40 if self.opt.dataset_name == "t2m" else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20) : int(to_tag * 20)]
                                if (len(n_motion)) < min_motion_len or (
                                    len(n_motion) >= 200
                                ):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                    + "_"
                                    + name
                                )
                                while new_name in data_dict:
                                    new_name = (
                                        random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                        + "_"
                                        + name
                                    )
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1])
        )

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4 : 4 + (joints_num - 1) * 3] = std[4 : 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3 : 4 + (joints_num - 1) * 9] = (
                std[4 + (joints_num - 1) * 3 : 4 + (joints_num - 1) * 9] / 1.0
            )
            # local_velocity (B, seq_len, joint_num*3)
            std[
                4 + (joints_num - 1) * 9 : 4 + (joints_num - 1) * 9 + joints_num * 3
            ] = (
                std[
                    4 + (joints_num - 1) * 9 : 4 + (joints_num - 1) * 9 + joints_num * 3
                ]
                / 1.0
            )
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3 :] = (
                std[4 + (joints_num - 1) * 9 + joints_num * 3 :] / opt.feat_bias
            )

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, "mean.npy"), mean)
            np.save(pjoin(opt.meta_dir, "std.npy"), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length: int) -> None:
        assert length <= self.opt.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        # print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.std + self.mean

    def __len__(self) -> int:
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item: int) -> _T_GET_ITEM:
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[: self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if self.opt.is_train:
            if m_length != self.max_length:
                if self.opt.unit_length < 10:
                    coin2 = np.random.choice(["single", "single", "double"])
                else:
                    coin2 = "single"
                if len_gap == 0 or (len_gap == 1 and coin2 == "double"):
                    m_length = self.max_length
                    idx = random.randint(0, m_length - self.max_length)
                    motion = motion[idx : idx + self.max_length]
                else:
                    if coin2 == "single":
                        n_m_length = self.max_length + self.opt.unit_length * len_gap
                    else:
                        n_m_length = self.max_length + self.opt.unit_length * (
                            len_gap - 1
                        )
                    idx = random.randint(0, m_length - n_m_length)
                    motion = motion[idx : idx + self.max_length]
                    m_length = n_m_length
        else:
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(["single", "single", "double"])
            else:
                coin2 = "single"

            if coin2 == "double":
                m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
            elif coin2 == "single":
                m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx : idx + m_length]

        # Z Normalization
        motion = (motion - self.mean) / self.std

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
        ), item


def list_matching_files(path):
    """
    For a hacky run
    """
    # Lists to store the filenames
    sampled_files = []
    sampled_and_imitated_files = []

    # Iterate over all files in the given path
    for file in os.listdir(path):
        if file.endswith("sampled.npy"):
            sampled_files.append(file)
        elif file.endswith("sampled_and_imitated.npy"):
            sampled_and_imitated_files.append(file)

    # Sort the lists
    sampled_files.sort()
    sampled_and_imitated_files.sort()
    print(len(sampled_files), len(sampled_and_imitated_files))

    # Find common prefixes
    keepers = []

    common_prefixes = []
    for file in sampled_files:
        prefix = file.replace("-sampled.npy", "")
        if f"{prefix}-sampled_and_imitated.npy" in sampled_and_imitated_files:
            common_prefixes.append(prefix)
            keepers.append(file)
            keepers.append(f"{prefix}-sampled_and_imitated.npy")

    # Sort the list of common prefixes
    common_prefixes.sort()

    keepers.sort()

    return keepers

IS_HACKY_RUN = False

if IS_HACKY_RUN:

    matches = list_matching_files("scripts/PALM/06_self_consuming_loop_v3/hacky/model/generation_1/synthetic_motions")
   
    import json
    matching_pairs_path = "scripts/PALM/06_self_consuming_loop_v3/hacky/model/generation_1/synthetic_motions/-computed_ones.json"   
    with open(matching_pairs_path, "w") as fp:
        json.dump(matches, fp, indent=4)


class Text2MotionDatasetV2(BaseDatasetMDM):
    """For use of training text motion matching model, and evaluations.

    Implements functions and loading for generated samples for autophagous training loops.

    NB: NOT a subclass of Text2MotionDataset!!!
    """

    def __init__(
        self,
        opt: Namespace,
        mean,
        std,
        split_file: str,
        w_vectorizer: WordVectorizer,
        subset_by_keyword: Optional[str] = None,
        synthetic_data_dir: Optional[str] = None,
        synthetic_augmentation_percent: Optional[float] = None,
        augmentation_type: Optional[str] = None,
        nearest_neighbor_POC_type: Optional[str] = None,
        is_fully_synthetic: Optional[bool] = False,
    ) -> None:
        super().__init__(std, mean)
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length: int = 20  # why is this hard-coded?
        self.pointer: int = 0
        self.max_motion_length: int = opt.max_motion_length
        self.save_generated_path = None
        self.motion_framerate: int = 20

        # --- setup ---
        self._min_motion_len: int = 40 if self.opt.dataset_name == "t2m" else 24
        self._max_motion_len: int = 200

        self.subset_by_keyword = subset_by_keyword

        data_dict: T_DATA_DICT = {}

        id_list = self.load_id_list_from_split_file(split_file)
        new_name_list: List[str] = []
        length_list: List[int] = []

        # data augmentation, for POC
        if synthetic_data_dir:

            # get list of available pre-saved synthesized motions
            prefixes_for_augmentation = []
            for filename in os.listdir(synthetic_data_dir):
                prefix = filename[:7] if "M" in filename[:7] else filename[:6]
                prefixes_for_augmentation.append(prefix)
            prefixes_for_augmentation = sorted(list(set(prefixes_for_augmentation)))

        # --- iterate through dataset loaded from disk ---
        num_skipped = 0
        num_synthetic_examples_included = 0

        for name in tqdm(id_list):
            try:

                # process the text data associated with the motion sequence
                lines = map(
                    self.parse_datapoint_line,
                    self.get_codec_lines(opt.text_dir, name),
                )

                if self.subset_by_keyword and self.subset_by_keyword == "NOKEYWORD":
                    pass  # in this case, we don't want to skip this one!!
                elif self.subset_by_keyword and not any(
                    (self.subset_by_keyword in x[0] for x in lines)
                ):
                    # skip this sample if it does NOT contain the given keyword
                    num_skipped += 1
                    continue

                # if using HumanML3D, the motion dir is "new_joint_vecs", not "joint_vecs"
                # "Extracted rotation invariant feature and rotation features vectors from 3d motion positions"
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if self._motion_is_bad_size(motion):
                    continue

                text_data = []
                flag = False

                # process the text data associated with the motion sequence
                lines = self.get_codec_lines(opt.text_dir, name)
                for line in lines:
                    # --- create text dict ---
                    caption, tokens, f_tag, to_tag = self.parse_datapoint_line(line)

                    text_dict: MotionTextSampleDict = {
                        "caption": caption,
                        "tokens": tokens,
                    }

                    if f_tag == to_tag == 0.0:
                        # use the entirety of the motion, skip the steps that
                        # isolate the segment corresponding to the caption
                        flag = True
                        text_data.append(text_dict)
                    else:
                        # isolate the segment corresponding to the caption
                        try:
                            n_motion, name = self._trim_and_hash_motion(
                                f_tag, to_tag, motion, caption, name
                            )
                            if self._motion_is_bad_size(n_motion):
                                continue

                            # add the sample to the data dictionary
                            motionSequenceSampleDict : MotionSequenceSampleDict = {
                                "motion": n_motion,
                                "length": len(n_motion),
                                "text": [text_dict],
                            }
                            data_dict[name] = motionSequenceSampleDict
                            new_name_list.append(name)
                            length_list.append(len(n_motion))

                        except Exception as e:
                            print(f"Exception: {e}, on line: {line}")


                if flag:
                    # no random id added as a prefix, just using the name assigned to
                    # the datapoint, ex. "000014", "M005677"
                    # if there is a prefix, it is the md5 hash of the motion length, text, and parent name
                    motionSequenceSampleDict : MotionSequenceSampleDict = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    data_dict[name] = motionSequenceSampleDict
                    new_name_list.append(name)
                    length_list.append(len(motion))

                    if synthetic_data_dir and name in prefixes_for_augmentation:

                        if augmentation_type:

                            if augmentation_type == "raw_mdm_output":
                                name_synthetic = f"{name}-sampled.npy"
                            elif augmentation_type == "imitation_output":
                                name_synthetic = f"{name}-sampled_and_imitated.npy"

                            motion_path = pjoin(synthetic_data_dir, name_synthetic)
                            motion_synthetic = np.load(motion_path)
                                
                        motionSequenceSampleDict = {
                            "motion": motion_synthetic,
                            "length": len(motion_synthetic),
                            "text": text_data,
                        }
                        data_dict[name_synthetic] = motionSequenceSampleDict

                        new_name_list.append(name_synthetic)
                        length_list.append(len(motion_synthetic))
                        num_synthetic_examples_included += 1
                        
            except:
                pass


        if is_fully_synthetic:
            # if we're running a fully synthetic loop, we want to just restrict to the subset
            # of examples we've included that are synthetic. the logic here manually post-processes
            # to remove all the non-synthetic examples, by editing new_name_list, length_list, data_dict

            names_to_keep = []
            idxs_to_keep = []
            for i, (name, _) in enumerate(zip(new_name_list, length_list)):

                if augmentation_type == "raw_mdm_output":
                    suffix = "-sampled.npy"
                elif augmentation_type == "imitation_output":
                    suffix = "-sampled-and-imitated.npy"
                else:
                    raise ValueError

                if suffix in name:
                    idxs_to_keep.append(i)
                    names_to_keep.append(name)
            
            new_name_list = [new_name_list[i] for i in idxs_to_keep]
            length_list = [length_list[i] for i in idxs_to_keep]

            data_dict = {key : data_dict[key] for key in names_to_keep}
            
        print(f"num_synthetic_examples_included = {num_synthetic_examples_included}")
        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1])
        )

        self.length_arr = np.array(length_list)  # len = 1820
        self.data_dict = data_dict  # len = 1664
        self.name_list: List[str] = name_list  # len = 1820
        self.reset_max_len(self.max_length)  # len = 20

        print(f"Skipped: {num_skipped} samples. Total Samples: {len(self.data_dict)}")

    @staticmethod
    def load_id_list_from_split_file(split_file: str) -> List[str]:
        id_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        return id_list

    def reset_max_len(self, length: int) -> None:
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        # print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def get_motion_sequence_dict_key_name(
        self, start_frame: int, end_frame: int, caption: str, parent_seq_name: str
    ) -> str:
        """
        Loading motion sequences from disk in these dataloaders associated motion sequences
        in a mapping like:
        (string key: (motion sequence data, sequence length, associated caption and caption's tokenization))

        The string key for subsequences of longer motion sequences was generated like this:

            new_name = (
                random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                    + "_"
                    + name
                )
                while new_name in data_dict:
                    new_name = (
                        random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                        + "_"
                        + name
                    )

        This is not a huge issue, but it removes determinism from data loading, which we would
        want for an autophagous loop that edits the data dictionary.
        """
        base = f"{start_frame}_{end_frame}_{parent_seq_name}_{caption}"
        return hashlib.md5(base.encode()).hexdigest() + "_" + parent_seq_name

    def _trim_and_hash_motion(
        self, f_tag: float, to_tag: float, motion: np.ndarray, caption: str, name: str
    ) -> Tuple[np.ndarray, str]:
        # all motions were downsampled to 20 fps

        # inclusive start lower bound
        start_frame = int(f_tag * self.motion_framerate)

        # not actually included in motion (exclusive upper bound)
        end_frame_excl = int(to_tag * self.motion_framerate)

        # trim it to include start frame, go right up to end frame
        n_motion = motion[start_frame:end_frame_excl]

        # deterministic hash of sequence for dictionary
        new_name = self.get_motion_sequence_dict_key_name(
            start_frame,
            end_frame_excl,
            caption,
            name,
        )

        # return the trimmed motion and its corresponding hash name
        return n_motion, new_name

    def _motion_is_bad_size(self, n_motion: np.ndarray) -> bool:
        return (len(n_motion)) < self._min_motion_len or (
            len(n_motion) >= self._max_motion_len
        )

    @staticmethod
    def get_codec_lines(text_dir: str, name: str) -> List[str]:
        rl = []
        with cs.open(pjoin(text_dir, name + ".txt")) as f:
            rl = f.readlines()
        return rl

    @staticmethod
    def parse_datapoint_line(line: str) -> Tuple[str, List[str], float, float]:
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

    def set_save_generated_path(self, save_generated_path: str) -> None:
        self.save_generated_path = save_generated_path

    def load_generated_samples_into_data_dict(
        self, from_epoch: int, args_hash: str
    ) -> None:
        if from_epoch < 0:
            # first run, do nothing
            return None

        if self.is_generated_samples_already_on_disk(from_epoch, args_hash):
            print(
                f"Detected already existing generated samples from end of epoch: {from_epoch} for model with args hash: {args_hash}."
            )
            # must load such samples and replace existing ones in memory
            to_replace = self.load_generated_samples_from_disk(from_epoch, args_hash)

            # in memory, given the keys we loaded from disk, replace those in
            # data_dict, using the (generated) samples instead of ground truth
            self.replace_samples(to_replace)
            print(
                f"Successfully replaced {len(to_replace)} previously generated samples in memory.\n"
            )

    def replace_samples(self, generated_sample_batch: T_DATA_DICT) -> None:
        """
        Edit the samples the model is using to train with those generated by the diffusion model.
        Assumes we use HumanML3D as our data set. Does NOT persist samples to disk, for that
        functionality, see save_generated_samples_to_disk
        """
        for datapoint_name, datapoint_data in generated_sample_batch.items():
            self.data_dict[datapoint_name] = datapoint_data

        print("Finished replacing loaded samples in memory.")

    def _get_generated_samples_filename(
        self, for_epoch: int, for_args_hash: str
    ) -> str:
        return "epoch_{}_gen_samps_{}.npy".format(for_epoch, for_args_hash)

    def _get_generated_samples_path(self, for_epoch: int, args_hash: str) -> str:
        if not self.save_generated_path:
            raise RuntimeError(
                "No generated sample save directory supplied! Set argument --save_generate_path when running."
            )
        return os.path.join(
            self.save_generated_path,
            self._get_generated_samples_filename(for_epoch, args_hash),
        )

    def is_generated_samples_already_on_disk(
        self, for_epoch: int, args_hash: str
    ) -> bool:
        save_file_path = self._get_generated_samples_path(for_epoch, args_hash)
        return os.path.exists(save_file_path)

    def save_generated_samples_to_disk(
        self, generated_samples: T_DATA_DICT, for_epoch: int, args_hash: str
    ) -> None:
        save_file_path = self._get_generated_samples_path(for_epoch, args_hash)

        # save as pickled dictionary / array object
        np.save(save_file_path, generated_samples)

        print(f"Saved samples to: {save_file_path}")

    def load_generated_samples_from_disk(
        self, for_epoch: int, args_hash: str
    ) -> T_DATA_DICT:
        save_file_path = self._get_generated_samples_path(for_epoch, args_hash)
        return load_pickled_np_dict(save_file_path)

    def get_name_from_item(self, item: int) -> str:
        idx = self.pointer + item
        return self.name_list[idx]

    def __len__(self) -> int:
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item: int) -> _T_GET_ITEM:
        """
        Collation later on does stuff to turn this single item fetching function to return
        single items in batches. Here, item is an integer, but objects from the data loader
        will be returned as a *tensor* of integer, with a size equal to the batch size.
        """
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[: self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []

        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])

        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length

        f_idx = random.randint(0, len(motion) - m_length)
        motion = motion[f_idx : f_idx + m_length]

        # Z Normalization
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [
                    motion,
                    np.zeros((self.max_motion_length - m_length, motion.shape[1])),
                ],
                axis=0,
            )

        # this object is referenced and transformed by t2m_collate:
        # data_loaders/tensors.py
        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            # before tokens are returned to an iterator, they are flattened into a list
            # instead of a list of lists, e.g.
            #
            #   ['sos/OTHER', 'the/DET', 'person/NOUN', 'is/AUX'] ->
            #   'sos/OTHER_the/DET_person/NOUN_is/AUX'
            #
            # BUT, this form is NOT how the tokens are stored in memory.
            "_".join(tokens),
        ), item


class Text2MotionDatasetBaseline(data.Dataset):
    """For use of training baseline"""

    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name == "t2m" else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20) : int(to_tag * 20)]
                                if (len(n_motion)) < min_motion_len or (
                                    len(n_motion) >= 200
                                ):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                    + "_"
                                    + name
                                )
                                while new_name in data_dict:
                                    new_name = (
                                        random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                        + "_"
                                        + name
                                    )
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1])
        )

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        # print("Dataset pointer pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self) -> int:
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item: int) -> _T_GET_ITEM:
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[: self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(["single", "single", "double"])
            else:
                coin2 = "single"
            if len_gap == 0 or (len_gap == 1 and coin2 == "double"):
                m_length = self.max_length
                s_idx = random.randint(0, m_length - self.max_length)
            else:
                if coin2 == "single":
                    n_m_length = self.max_length + self.opt.unit_length * len_gap
                else:
                    n_m_length = self.max_length + self.opt.unit_length * (len_gap - 1)
                s_idx = random.randint(0, m_length - n_m_length)
                m_length = n_m_length
        else:
            s_idx = 0

        src_motion = motion[s_idx : s_idx + m_length]
        tgt_motion = motion[s_idx : s_idx + self.max_length]

        # Z Normalization
        src_motion = (src_motion - self.mean) / self.std
        tgt_motion = (tgt_motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            src_motion = np.concatenate(
                [
                    src_motion,
                    np.zeros((self.max_motion_length - m_length, motion.shape[1])),
                ],
                axis=0,
            )
        return (
            word_embeddings,
            caption,
            sent_len,
            src_motion,
            tgt_motion,
            m_length,
        ), item


class MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4 : 4 + (joints_num - 1) * 3] = std[4 : 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3 : 4 + (joints_num - 1) * 9] = (
                std[4 + (joints_num - 1) * 3 : 4 + (joints_num - 1) * 9] / 1.0
            )
            # local_velocity (B, seq_len, joint_num*3)
            std[
                4 + (joints_num - 1) * 9 : 4 + (joints_num - 1) * 9 + joints_num * 3
            ] = (
                std[
                    4 + (joints_num - 1) * 9 : 4 + (joints_num - 1) * 9 + joints_num * 3
                ]
                / 1.0
            )
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3 :] = (
                std[4 + (joints_num - 1) * 9 + joints_num * 3 :] / opt.feat_bias
            )

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, "mean.npy"), mean)
            np.save(pjoin(opt.meta_dir, "std.npy"), std)

        self.mean = mean
        self.std = std
        print(
            "Total number of motions {}, snippets {}".format(
                len(self.data), self.cumsum[-1]
            )
        )

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item: int) -> _T_GET_ITEM:
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx : idx + self.opt.window_size]
        # Z Normalization
        motion = (motion - self.mean) / self.std

        return motion, item


class RawTextDataset(data.Dataset):
    def __init__(self, opt, mean, std, text_file, w_vectorizer):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.nlp = spacy.load("en_core_web_sm")

        with cs.open(text_file) as f:
            for line in f.readlines():
                word_list, pos_list = self.process_text(line.strip())
                tokens = [
                    "%s/%s" % (word_list[i], pos_list[i]) for i in range(len(word_list))
                ]
                self.data_dict.append({"caption": line.strip(), "tokens": tokens})

        self.w_vectorizer = w_vectorizer
        print("Total number of descriptions {}".format(len(self.data_dict)))

    def process_text(self, sentence):
        sentence = sentence.replace("-", "")
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == "NOUN" or token.pos_ == "VERB") and (word != "left"):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[item]
        caption, tokens = data["caption"], data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[: self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len


class TextOnlyDataset(data.Dataset):
    def __init__(
        self, opt, mean, std, split_file, subset_by_keyword: Optional[str] = None
    ):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 120
        self.subset_by_keyword = subset_by_keyword

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                    + "_"
                                    + name
                                )
                                while new_name in data_dict:
                                    new_name = (
                                        random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                        + "_"
                                        + name
                                    )
                                data_dict[new_name] = {"text": [text_dict]}
                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {"text": text_data}
                    new_name_list.append(name)
            except:
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self) -> int:
        return len(self.data_dict)

    def __getitem__(self, item: int) -> _T_GET_ITEM:
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        text_list = data["text"]

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        # fixed_length can be set from outside before sampling
        return (None, None, caption, None, np.array([0]), self.fixed_length, None), item


class HumanML3D(data.Dataset):
    """
    A wrapper class for t2m original dataset for MDM purposes
    """

    def __init__(
        self,
        mode: T_HUMANML3D_KIT_DATASET_MODE,
        datapath: str = "./dataset/humanml_opt.txt",
        split: T_HUMANML3D_KIT_DATASET_SPLIT_TYPE = "train",
        **kwargs,
    ) -> None:
        self.mode = mode
        self.num_frames: int = kwargs.get("num_frames", 0)

        if self.num_frames == 0:
            raise ValueError(
                "Provide a num_frames argument to define the maximum number of frames a sequence in the dataset can have"
            )

        self.dataset_name = "t2m"
        self.dataname = "t2m"

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f"."
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = (
            None  # torch.device('cuda:4') # This param is not in use in this context
        )
        opt: Namespace = get_opt(dataset_opt_path, device, **kwargs)

        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)  # i.e. ./dataset/HumanML3D
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = "./dataset"
        self.opt = opt
        print("Loading dataset %s ..." % opt.dataset_name)

        if mode == "gt":
            # used by T2M models (including evaluators)
            self.mean = np.load(pjoin(opt.meta_dir, f"{opt.dataset_name}_mean.npy"))
            self.std = np.load(pjoin(opt.meta_dir, f"{opt.dataset_name}_std.npy"))
        elif mode in ["train", "eval", "text_only"]:
            # used by our models
            self.mean = np.load(pjoin(opt.data_root, "Mean.npy"))
            self.std = np.load(pjoin(opt.data_root, "Std.npy"))

        if mode == "eval":
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            self.mean_for_eval = np.load(
                pjoin(opt.meta_dir, f"{opt.dataset_name}_mean.npy")
            )
            self.std_for_eval = np.load(
                pjoin(opt.meta_dir, f"{opt.dataset_name}_std.npy")
            )

        # path to the split of the HumanML3D dataset, e.g. ./dataset/HumanML3D/train.txt
        self.split_file = pjoin(opt.data_root, f"{split}.txt")

        self.subset_by_keyword = kwargs.get("subset_by_keyword", None)

        if mode == "text_only":
            self.t2m_dataset = TextOnlyDataset(
                self.opt, self.mean, self.std, self.split_file, self.subset_by_keyword
            )
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, "glove"), "our_vab")
            self.t2m_dataset = Text2MotionDatasetV2(
                self.opt,
                self.mean,
                self.std,
                self.split_file,
                self.w_vectorizer,
                self.subset_by_keyword,
                synthetic_data_dir=kwargs.get("synthetic_data_dir", None),
                synthetic_augmentation_percent=kwargs.get(
                    "synthetic_augmentation_percent", None
                ),
                augmentation_type=kwargs.get("augmentation_type", None),
                nearest_neighbor_POC_type=kwargs.get("nearest_neighbor_POC_type", None),
                is_fully_synthetic=kwargs.get("is_fully_synthetic", None),
            )
            self.mean = self.t2m_dataset.mean
            self.std = self.t2m_dataset.std
            self.num_actions = 1  # dummy placeholder

        assert len(self.t2m_dataset) > 1, (
            "You loaded an empty dataset, "
            "it is probably because your data dir has only texts and no motions.\n"
            "To train and evaluate MDM you should get the FULL data as described "
            "in the README file."
        )

    def __getitem__(self, item: int) -> _T_GET_ITEM:
        return self.t2m_dataset.__getitem__(item)

    def __len__(self) -> int:
        return self.t2m_dataset.__len__()


class KIT(HumanML3D):
    """
    A wrapper class for t2m original dataset for MDM purposes
    """

    def __init__(
        self,
        mode: T_HUMANML3D_KIT_DATASET_MODE,
        datapath: str = "./dataset/kit_opt.txt",
        split: str = "train",
        **kwargs,
    ) -> None:
        super(KIT, self).__init__(mode, datapath, split, **kwargs)
