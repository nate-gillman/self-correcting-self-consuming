import contextlib
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        from uhc.data_process.process_amass_db import *
        from scripts.eval_uhc import parse_args
        from visualize_chains import visualize_samples_and_save_to_disk
        from uhc.agents.agent_copycat import AgentCopycat
    except:
        from UniversalHumanoidControl.uhc.data_process.process_amass_db import *
        from UniversalHumanoidControl.scripts.eval_uhc import parse_args
        from UniversalHumanoidControl.visualize_chains import visualize_samples_and_save_to_disk
        from UniversalHumanoidControl.uhc.agents.agent_copycat import AgentCopycat

SMPL_MODEL_PATH = "UniversalHumanoidControl/data/smpl"


def fix_height_smpl_vanilla(pose_aa, th_trans, th_betas, gender, seq_name):
    # no filtering, just fix height
    gender = gender.item() if isinstance(gender, np.ndarray) else gender
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")

    if gender == "neutral":
        with contextlib.redirect_stdout(None):
            # silence warning about 10 beta shape coefficient
            smpl_parser = SMPL_Parser(model_path=SMPL_MODEL_PATH, gender="neutral", use_pca=False,
                                      create_transl=False)
    elif gender == "male":
        smpl_parser = smpl_parser_m
    elif gender == "female":
        smpl_parser = smpl_parser_f
    else:
        print(gender)
        raise Exception("Gender Not Supported!!")

    batch_size = pose_aa.shape[0]
    verts, jts = smpl_parser.get_joints_verts(pose_aa[0:1], th_betas.repeat((1, 1)), th_trans=th_trans[0:1])

    # vertices = verts[0].numpy()
    gp = torch.min(verts[:, :, 2])

    # if gp < 0:
    th_trans[:, 2] -= gp

    return th_trans


def process_qpos_list(qpos_dict):
    amass_res_batch = {}
    removed_k = []
    pbar = qpos_dict

    for k, v in qpos_dict.items():
        # print("=" * 20)
        k = "0-" + k
        seq_name = k
        betas = np.zeros(10)  # v["betas"]
        gender = "neutral"  # v["gender"]
        amass_fr = 60.0  # v["mocap_framerate"]
        target_fr = 60.0
        skip = int(amass_fr / target_fr)
        amass_pose = v["poses"][::skip]
        amass_trans = v["trans"][::skip]

        bound = amass_pose.shape[0]
        # if k in amass_occlusion:
        #     issue = amass_occlusion[k]["issue"]
        #     if (issue == "sitting" or issue == "airborne") and "idxes" in amass_occlusion[k]:
        #         bound = amass_occlusion[k]["idxes"][0]  # This bounded is calucaled assuming 30 FPS.....
        #         if bound < 10:
        #             print("bound too small", k, bound)
        #             continue
        #     else:
        #         print("issue irrecoverable", k, issue)
        #         continue

        seq_length = amass_pose.shape[0]
        # if seq_length < 10:
        #     continue
        with torch.no_grad():
            pose_aa = torch.tensor(amass_pose)[:bound]  # After sampling the bound
            amass_trans = torch.tensor(amass_trans[:bound])  # After sampling the bound
            betas = torch.from_numpy(betas)
            batch_size = pose_aa.shape[0]

            # amass_trans = fix_height_smpl(
            #     pose_aa=pose_aa,
            #     th_betas=betas,
            #     th_trans=amass_trans,
            #     gender=gender,
            #     seq_name=k,
            # )
            # if amass_trans is None:
            #     removed_k.append(k)
            #     continue

            amass_trans = fix_height_smpl_vanilla(
                pose_aa=pose_aa,
                th_betas=betas,
                th_trans=amass_trans,
                gender=gender,
                seq_name=k,
            )

            pose_seq_6d = convert_aa_to_orth6d(torch.tensor(pose_aa)).reshape(batch_size, -1, 6)

            amass_res_batch[seq_name] = {
                "pose_aa": pose_aa.numpy(),
                "pose_6d": pose_seq_6d.numpy(),
                # "qpos": qpos,
                "trans": amass_trans.numpy(),
                "beta": betas.numpy(),
                "seq_name": seq_name,
                "gender": gender,
            }

    return amass_res_batch


def get_imitation(amass_res):
    args, unknown_args = parse_args()
    # OPTIONS: ["uhc_explicit", "uhc_implicit", "uhc_implicit_shape"]
    try:
        args.cfg = unknown_args[unknown_args.index('--uhc_model_option') + 1]
    except:
        print("DEFAULTING TO: args.cfg = 'uhc_explicit'")
        args.cfg = "uhc_explicit"
    print("uhc_model_option:", args.cfg)
    args.epoch = {"uhc_explicit": 5000, "uhc_implicit": 19000, "uhc_implicit_shape": 4700}[args.cfg]
    args.mode = "stats"
    args.data = "sample_data/amass_copycat_take5_test_small.pkl"

    cfg = Config(cfg_id=args.cfg, create_dirs=False)
    cfg.update(args)

    # quit()
    # cfg.output = osp.join("results/renderings/uhc/", f"{cfg.id}")
    # os.makedirs(cfg.output, exist_ok=True)

    cfg.data_specs["file_path"] = args.data
    if "test_file_path" in cfg.data_specs:
        del cfg.data_specs["test_file_path"]

    cfg.data_specs["preprocessed_motion"] = amass_res

    # if cfg.mode == "vis":
    #     cfg.num_threads = 1

    dtype = torch.float64
    torch.set_default_dtype(dtype)

    device = torch.device("cuda:0")
    print(f"Using: {device}")

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # if cfg.smplx and cfg.robot_cfg["model"] == "smplh":
    #     cfg.robot_cfg["model"] = "smplx"

    # keys() = [
    #   'gt', 'pred', 'gt_jpos', 'pred_jpos', 'vf_world', 'reward', 'percent',
    #   'fail_safe', 'root_dist', 'pa_mpjpe', 'mpjpe', 'mpjpe_g', 'accel_dist', 'vel_dist', 'succ']

    # print("HERE IS A BIT TIME CONSUMING... PROBABLY CUZ I/O OR MODEL LOADING") 
    agent = AgentCopycat(cfg, dtype, device, training=True, checkpoint_epoch=args.epoch)
    res_dicts = agent.eval_policy(epoch=args.epoch, dump=False)

    return res_dicts[0]


def amass_to_imitation(amass_data_batch):
    # STEP 1: uhc/data_process/process_amass_raw.py; I'd change this to a dict w one entry for each clip to do it in batches...
    db = {}
    for i in range(len(amass_data_batch)):
        db[f"motion_seq_{i:06d}"] = amass_data_batch[i]
        # print(amass_data["poses"][0]) # 372, 72

    # STEP 2: uhc/data_process/process_amass_db.py
    amass_res = process_qpos_list(db)
    # amass_res["0-motion_seq"].keys() = ['pose_aa', 'pose_6d', 'trans', 'beta', 'seq_name', 'gender']
    # NOTE: input and output trans are off SLIGHTLY; if we want, we can FIX THE OUTPUT ONE... MIGHT HELP W PERFORMANCE???
    # print(amass_res["0-motion_seq"]["pose_aa"][0]) # 372, 72

    # STEP 3: pass it through imitation pipeline...
    # gt_jpos, pred_jpos, gt, pred, gt_pose_aa, pred_body_quat = get_imitation(amass_res)
    imitations_batch = get_imitation(amass_res)

    return imitations_batch["imitation_of_synthetic_motions"]


def main():
    # load in amass data
    amass_data = np.load(
        "/oscar/data/csun45/nates_stuff/motion-diffusion-model/translation/data/amass/CMU_SMPL_HG/75_09_poses.npz")
    amass_data = {
        "poses": amass_data["poses"][:, :72],  # (372, 72)
        "trans": amass_data["trans"],  # (372, 3)
    }
    # kinematic_chain = [ # original...
    #     [0, 2, 5, 8, 11],         # pelvis, right hip, right knee, right ankle, right foot
    #     [0, 1, 4, 7, 10],         # pelvis, left hip, left knee, left ankle, left foot
    #     [0, 3, 6, 9, 12, 15],     # pelvis, spine1, spine2, spine3, neck, head
    #     [9, 14, 17, 19, 21],      # spine3, right collar, right shoulder, right elbow, right wrist
    #     [9, 13, 16, 18, 20]       # spine3, left collar, left shoulder, left elbow, left wrist
    # ]

    # imitate it
    gt_jpos, pred_jpos, _, _ = amass_to_imitation(amass_data)  # (372, 72)
    # visualize imitated motion
    kinematic_chain = [
        [0, 5, 6, 7, 8],  # pelvis, left hip, left knee, left ankle, left foot
        [0, 1, 2, 3, 4],  # pelvis, right hip, right knee, right ankle, right foot
        [0, 9, 10, 11, 12, 13],  # pelvis, spine1, spine2, spine3, neck, head
        [11, 19, 20, 21, 22],  # spine3, left collar, left shoulder, left elbow, left wrist
        [11, 14, 15, 16, 17]  # spine3, right collar, right shoulder, right elbow, right wrist
    ]
    visualize_samples_and_save_to_disk(
        gt_jpos,
        "/oscar/data/csun45/txt2vid/mdm/motion-diffusion-model/visualize/amass_to_imitation",
        "vid_gt.mp4",
        60,
        kinematic_chain
    )
    visualize_samples_and_save_to_disk(
        pred_jpos,
        "/oscar/data/csun45/txt2vid/mdm/motion-diffusion-model/visualize/amass_to_imitation",
        "vid_pred.mp4",
        60,
        kinematic_chain
    )

    return None


if __name__ == "__main__":
    main()
