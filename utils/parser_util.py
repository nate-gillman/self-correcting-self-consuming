import argparse
import json
import os
from argparse import ArgumentParser, Namespace
from typing import List

from typing_extensions import Literal


def parse_and_load_from_model(parser: ArgumentParser) -> Namespace:
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    return overwrite_args_from_argparser(args, parser)


def overwrite_args_from_argparser(args, parser):
    args_to_overwrite = []
    for group_name in ["dataset", "model", "diffusion"]:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    try: # will take this path during sampling+rendering
        model_path = args.model_path
        args_path = os.path.join(os.path.dirname(model_path), "args.json")
    except: # will take this path during training+sampling
        model_path = get_model_path_from_args() or args.model_path
        args_path = os.path.join(os.path.dirname(model_path), "args.json")
    if not os.path.exists(args_path):
        args_path = os.path.join(model_path, "args.json")
    assert os.path.exists(args_path), f"Arguments json file was not found! Expected args.json at location: {args_path}"
    with open(args_path, "r") as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif "cond_mode" in model_args:  # backward compitability
            unconstrained = model_args["cond_mode"] == "no_cond"

            # unconstrained is true if {"cond_mode": "no_cond"} is in the args
            setattr(args, "unconstrained", unconstrained)

        else:
            print(
                "Warning: was not able to load [{}], using default value [{}] instead.".format(
                    a, args.__dict__[a]
                )
            )

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args


def get_args_per_group_name(
    parser: ArgumentParser, args: Namespace, group_name: str
) -> List[str]:
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {
                a.dest: getattr(args, a.dest, None) for a in group._group_actions
            }
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError("group_name was not found.")


def get_model_path_from_args() -> str:
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument("model_path")
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        # raise ValueError("model_path argument must be specified.")
        print("WARNING: --model_path was not specified through args.")
        return ""


def add_base_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("base")
    group.add_argument(
        "--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU."
    )
    group.add_argument(
        "--is_self_consuming",
        default=False,
        type=bool,
        help="Use the self-consuming loop.",
    )
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument(
        "--no_fixseed",
        action="store_true",
        help="Add this flag if you do not want to fix the random seed to the value of --seed (which is 10 by default)",
    )
    group.add_argument(
        "--batch_size", default=64, type=int, help="Batch size during training."
    )
    group.add_argument(
        "--use_wandb",
        default=0,
        type=int,
        help="Use this argument if logging to wandb",
    )
    group.add_argument(
        "--filter_dataset_by_keyword",
        default="",
        type=str,
        help="The word that must appear in the caption of data to be used in training.",
    )


def add_diffusion_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("diffusion")
    group.add_argument(
        "--noise_schedule",
        default="cosine",
        choices=["linear", "cosine"],
        type=str,
        help="Noise schedule type",
    )
    group.add_argument(
        "--diffusion_steps",
        default=1000,
        type=int,
        help="Number of diffusion steps (denoted T in the paper)",
    )
    group.add_argument(
        "--sigma_small", default=True, type=bool, help="Use smaller sigma values."
    )


def add_model_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("model")
    group.add_argument(
        "--arch",
        default="trans_enc",
        choices=["trans_enc", "trans_dec", "gru"],
        type=str,
        help="Architecture types as reported in the paper.",
    )
    group.add_argument(
        "--emb_trans_dec",
        default=False,
        type=bool,
        help="For trans_dec architecture only, if true, will inject condition as a class token"
        " (in addition to cross-attention).",
    )
    group.add_argument("--layers", default=8, type=int, help="Number of layers.")
    group.add_argument(
        "--latent_dim", default=512, type=int, help="Transformer/GRU width."
    )
    group.add_argument(
        "--cond_mask_prob",
        default=0.1,
        type=float,
        help="The probability of masking the condition during training."
        " For classifier-free guidance learning.",
    )
    group.add_argument(
        "--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss."
    )
    group.add_argument(
        "--lambda_vel", default=0.0, type=float, help="Joint velocity loss."
    )
    group.add_argument(
        "--lambda_fc", default=0.0, type=float, help="Foot contact loss."
    )
    group.add_argument(
        "--unconstrained",
        action="store_true",
        help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
        "Currently tested on HumanAct12 only.",
    )


def add_data_options(parser) -> None:
    group = parser.add_argument_group("dataset")
    group.add_argument(
        "--dataset",
        default="humanml",
        choices=["humanml", "kit", "humanact12", "uestc"],
        type=str,
        help="Dataset name (choose from list).",
    )
    group.add_argument(
        "--data_dir",
        default="",
        type=str,
        help="If empty, will use defaults according to the specified dataset.",
    )


def add_training_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("training")
    group.add_argument(
        "--save_dir",
        required=True,
        type=str,
        help="Path to save checkpoints and results.",
    )
    group.add_argument(
        "--overwrite",
        action="store_true",
        help="If True, will enable to use an already existing save_dir.",
    )
    group.add_argument(
        "--train_platform_type",
        default="NoPlatform",
        choices=["NoPlatform", "ClearmlPlatform", "TensorboardPlatform"],
        type=str,
        help="Choose platform to log results. NoPlatform means no logging.",
    )
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument(
        "--weight_decay", default=0.0, type=float, help="Optimizer weight decay."
    )
    group.add_argument(
        "--lr_anneal_steps",
        default=0,
        type=int,
        help="Number of learning rate anneal steps.",
    )
    group.add_argument(
        "--eval_batch_size",
        default=32,
        type=int,
        help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
        "T2m precision calculation is based on fixed batch size 32.",
    )
    group.add_argument(
        "--eval_split",
        default="test",
        choices=["val", "test"],
        type=str,
        help="Which split to evaluate on during training.",
    )
    group.add_argument(
        "--eval_during_training",
        action="store_true",
        help="If True, will run evaluation during training.",
    )
    group.add_argument(
        "--eval_rep_times",
        default=3,
        type=int,
        help="Number of repetitions for evaluation loop during training.",
    )
    group.add_argument(
        "--eval_num_samples",
        default=1_000,
        type=int,
        help="If -1, will use all samples in the specified split.",
    )
    group.add_argument(
        "--log_interval", default=1000, type=int, help="Log losses each N steps"
    )
    group.add_argument(
        "--log_interval_wandb", default=10, type=int, help="Log losses each N steps"
    )
    group.add_argument(
        "--save_interval",
        default=25_000,
        type=int,
        help="Save checkpoints and run evaluation each N steps",
    )
    group.add_argument(
        "--num_steps",
        default=600_000,
        type=int,
        help="Training will stop after the specified number of steps.",
    )
    group.add_argument(
        "--num_frames",
        default=60,
        type=int,
        help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.",
    )
    group.add_argument(
        "--resume_checkpoint",
        default="",
        type=str,
        help="If not empty, will start from the specified checkpoint (path to model###.pt file).",
    )


def add_sampling_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("sampling")
    group.add_argument(
        "--model_path",
        required=False,
        type=str,
        help="Path to model####.pt file to be sampled.",
    )
    group.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="Path to results dir (auto created by the script). "
        "If empty, will create dir in parallel to checkpoint.",
    )
    group.add_argument(
        "--num_samples",
        default=10,
        type=int,
        help="Maximal number of prompts to sample, "
        "if loading dataset from file, this field will be ignored.",
    )
    group.add_argument(
        "--num_repetitions",
        default=3,
        type=int,
        help="Number of repetitions, per sample (text prompt/action)",
    )
    group.add_argument(
        "--guidance_param",
        default=2.5,
        type=float,
        help="For classifier-free sampling - specifies the s parameter, as defined in the paper.",
    )


def add_generate_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("generate")
    group.add_argument(
        "--motion_length",
        default=6.0,
        type=float,
        help="The length of the sampled motion [in seconds]. "
        "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)",
    )
    group.add_argument(
        "--input_text",
        default="",
        type=str,
        help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.",
    )
    group.add_argument(
        "--action_file",
        default="",
        type=str,
        help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
        "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
        "If no file is specified, will take action names from dataset.",
    )
    group.add_argument(
        "--text_prompt",
        default="",
        type=str,
        help="A text prompt to be generated. If empty, will take text prompts from dataset.",
    )
    group.add_argument(
        "--action_name",
        default="",
        type=str,
        help="An action name to be generated. If empty, will take text prompts from dataset.",
    )
    group.add_argument(
        "--is_generate_POC_samples",
        default=False,
        type=bool,
        help="If we want to generate a bunch of samples for the proof of concept",
    )
    group.add_argument(
        "--is_generate_batched_text_prompt_samples",
        default=False,
        type=bool,
        help="If we want to generate batched prompts (1 row per text prompt) the proof of concept",
    )


def add_edit_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("edit")
    group.add_argument(
        "--edit_mode",
        default="in_between",
        choices=["in_between", "upper_body"],
        type=str,
        help="Defines which parts of the input motion will be edited.\n"
        "(1) in_between - suffix and prefix motion taken from input motion, "
        "middle motion is generated.\n"
        "(2) upper_body - lower body joints taken from input motion, "
        "upper body is generated.",
    )
    group.add_argument(
        "--text_condition",
        default="",
        type=str,
        help="Editing will be conditioned on this text prompt. "
        "If empty, will perform unconditioned editing.",
    )
    group.add_argument(
        "--prefix_end",
        default=0.25,
        type=float,
        help="For in_between editing - Defines the end of input prefix (ratio from all frames).",
    )
    group.add_argument(
        "--suffix_start",
        default=0.75,
        type=float,
        help="For in_between editing - Defines the start of input suffix (ratio from all frames).",
    )


def add_evaluation_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("eval")
    group.add_argument(
        "--model_path",
        required=False,
        type=str,
        help="Path to model####.pt file to be sampled.",
    )
    group.add_argument(
        "--eval_mode",
        default="wo_mm",
        choices=["wo_mm", "mm_short", "debug", "full"],
        type=str,
        help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
        "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
        "debug - short run, less accurate results."
        "full (a2m only) - 20 repetitions.",
    )
    group.add_argument(
        "--guidance_param",
        default=2.5,
        type=float,
        help="For classifier-free sampling - specifies the s parameter, as defined in the paper.",
    )
    group.add_argument(
        "--override_num_diffusion_steps",
        default=0,
        type=int,
        help="Even if a model was trained on n diffusion steps, use this argument to override it during eval. If left at 0, use args.diffusion_steps",
    )


def add_autophagous_options(parser: ArgumentParser) -> None:
    """
    Important: any argument that will influence network outcomes should go here. Reason being that samples
    and their location on disk are named based on the hash of these arguments.

    Arguments that do not influence network outcomes, i.e. whether to use wandb or not, should NOT be
    put into the arguments namespace. Reason being that the argument has no influence on the network, so
    the samples it generates should not have a different hash than the same network with use_wandb=True
    just for example.
    """
    group = parser.add_argument_group("autophagous")

    group.add_argument(
        "--save_generated_path",
        type=str,
        default="dataset/Generated",
        help="The path to which generated samples will be saved during self consuming loop.",
    )
    group.add_argument(
        "--percent_samples_to_replace_per_epoch",
        type=float,
        # really small just for testing purposes
        default=0.002,
        help="The percentage of samples from the dataset to replace with generated samples per epoch.",
    )
    # group.add_argument(
    #     "--guidance_param",
    #     default=2.5,
    #     type=float,
    #     help="For classifier-free sampling - specifies the s parameter, as defined in the paper.",
    # )
    group.add_argument(
        "--wandb_project_name",
        default="",
        type=str,
        help="The name of the wandb project within which to log data.",
    )
    group.add_argument(
        "--wandb_exp_name",
        default="",
        type=str,
        help="The name of the wandb experiment you'll see in the UI",
    )
    group.add_argument(
        "--self_consuming_loop_freq",
        default=0,
        type=int,
        help="How many steps between synthesizing motions for the self consuming loop"
    )
    group.add_argument(
        "--synthetic_data_dir_parent",
        default="",
        type=str,
        help="Parent directory where all synthetic data subdirectories will be stored"
    )
    group.add_argument(
        "--uhc_model_option",
        default='uhc_explicit',
        type=str,
        choices=['uhc_explicit', 'uhc_implicit', 'uhc_implicit_shape']
    )
    group.add_argument(
        "--num_generations",
        default=0,
        type=int
    )
    group.add_argument(
        "--start_at_generation",
        default=0,
        type=int
    )
    group.add_argument(
        "--generation",
        default=0,
        type=int
    )
    group.add_argument(
        "--is_fully_synthetic",
        action="store_true",
        help="Add this flag if you dont want to AUGMENT, but instead want to REPLACE",
    )
    group.add_argument(
        "--mini_dataset_dir",
        default="",
        type=str
    )
    group.add_argument(
        "--ONLY_SYNTHESIZE_EXAMPLES",
        default=False,
        action="store_true"
    )



def get_cond_mode(args: Namespace) -> Literal["no_cond", "text", "action"]:
    if args.unconstrained:
        cond_mode = "no_cond"
    elif args.dataset in ["kit", "humanml"]:
        cond_mode = "text"
    else:
        cond_mode = "action"
    return cond_mode


def train_args() -> Namespace:
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def add_proof_of_concept_options(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("proof_of_concept")
    group.add_argument(
        "--synthetic_data_dir",
        default="",
        type=str,
        help="Directory where synthetic data was pre-computed.",
    )
    group.add_argument(
        "--synthetic_augmentation_percent",
        default=0.0,
        type=float,
        help="What percentage of the total number of training examples we should augment using.",
    )
    group.add_argument(
        "--augmentation_type",
        default="",
        choices=["raw_mdm_output", "imitation_output"],
        type=str,
    )

    group.add_argument(
        "--nearest_neighbor_POC_type",
        default="",
        choices=[
            "A_train_include",
            "B_train_include_with_random_from_train_include_and_train_exclude", 
            "C_train_include_with_nearest_neighbor_in_train_exclude_to_synthesized",
            "D_train_include_with_UHC_imitation_augmentation", 
            "E_train_include_with_train_exclude",
            "F_train_include_with_raw_mdm_output_augmentation", 
        ],
        type=str,
    )


def autophagous_train_args() -> Namespace:
    # these arguments required for training
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    add_proof_of_concept_options(parser)

    # our custom experiment arguments
    add_autophagous_options(parser)

    return parser.parse_args()


def generate_args() -> Namespace:
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)

    add_autophagous_options(parser)
    add_proof_of_concept_options(parser)
    add_training_options(parser)

    args = parse_and_load_from_model(parser)
    cond_mode = get_cond_mode(args)

    if (args.input_text or args.text_prompt) and cond_mode != "text":
        raise Exception(
            "Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name."
        )
    elif (args.action_file or args.action_name) and cond_mode != "action":
        raise Exception(
            "Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt."
        )

    return args


def edit_args() -> Namespace:
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser() -> Namespace:
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    args = parse_and_load_from_model(parser)
    cond_mode = get_cond_mode(args)
    args.cond_mode = cond_mode
    return args


def get_args_save_path(args: Namespace) -> str:
    # dump the training args to file
    return os.path.join(args.save_dir, "args.json")


def get_json_of_args(args: Namespace) -> str:
    return json.dumps(vars(args), indent=4, sort_keys=True)
