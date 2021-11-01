import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import os.path as osp
import cv2
import torch
import joblib
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from lib.core.config import FTHE_DATA_DIR, parse_args
from lib.data_utils.img_utils import split_into_chunks, get_chunk_with_overlap
from lib.data_utils.kp_utils import convert_kps
from lib.models import FTHE
from lib.models.smpl import SMPL_MODEL_DIR, SMPL, H36M_TO_J14
from lib.utils.demo_utils import convert_crop_cam_to_orig_img, images_to_video
from lib.utils.eval_utils import compute_accel, compute_error_accel, batch_compute_similarity_transform_torch, compute_error_verts, compute_errors, plot_accel
from lib.utils.slerp_filter_utils import quaternion_from_matrix, quaternion_slerp, quaternion_matrix
from lib.utils.renderer import Renderer


def get_sequence(start_index, end_index, seqlen=16):
    if start_index != end_index:
        return [i for i in range(start_index, end_index+1)]
    else:
        return [start_index for _ in range(seqlen)]


""" Smoothing codes from MEVA (https://github.com/ZhengyiLuo/MEVA) """
def quat_correct(quat):
    """ Converts quaternion to minimize Euclidean distance from previous quaternion (wxyz order) """
    for q in range(1, quat.shape[0]):
        if np.linalg.norm(quat[q-1] - quat[q], axis=0) > np.linalg.norm(quat[q-1] + quat[q], axis=0):
            quat[q] = -quat[q]
    return quat


def quat_smooth(quat, ratio = 0.3):
    """ Converts quaternion to minimize Euclidean distance from previous quaternion (wxyz order) """
    for q in range(1, quat.shape[0]):
        quat[q] = quaternion_slerp(quat[q-1], quat[q], ratio)
    return quat


def smooth_pose_mat(pose, ratio = 0.3):
    quats_all = []
    for j in range(pose.shape[1]):
        quats = []
        for i in range(pose.shape[0]):
            R = pose[i,j,:,:]
            quats.append(quaternion_from_matrix(R))
        quats = quat_correct(np.array(quats))
        quats = quat_smooth(quats, ratio = ratio)
        quats_all.append(np.array([quaternion_matrix(i)[:3,:3] for i in quats]))

    quats_all = np.stack(quats_all, axis=1)
    return quats_all


if __name__ == "__main__":
    cfg, cfg_file, args = parse_args()
    SMPL_MAJOR_JOINTS = np.array([1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21])
    device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


    """ Evaluation Options """
    target_dataset = args.dataset  # 'mpii3d' '3dpw'
    set = 'test'
    target_action = args.seq
    render = args.render
    render_plain = args.render_plain
    only_img = False
    render_frame_start = args.frame
    plot = args.plot
    avg_filter = args.filter
    gender = 'neutral'

    model = FTHE(
        n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        seqlen=cfg.DATASET.SEQLEN,
        hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
        pretrained=cfg.TRAIN.PRETRAINED_REGRESSOR,
        add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
        bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
        use_residual=cfg.MODEL.TGRU.RESIDUAL,
    ).to(cfg.DEVICE)

    if cfg.TRAIN.PRETRAINED != '' and os.path.isfile(cfg.TRAIN.PRETRAINED):
        print(f"==> Loaded pretrained model from {cfg.TRAIN.PRETRAINED}...")
        checkpoint = torch.load(cfg.TRAIN.PRETRAINED)
        best_performance = checkpoint['performance']
        model.load_state_dict(checkpoint['gen_state_dict'])
    else:
        print(f"{cfg.TRAIN.PRETRAINED} is not a pretrained model! Exiting...")
        import sys; sys.exit()

    model.regressor.smpl = SMPL(
        SMPL_MODEL_DIR,
        batch_size=64,
        create_transl=False,
        gender=gender
    ).cuda()
    dtype = torch.float
    J_regressor = torch.from_numpy(np.load(osp.join(FTHE_DATA_DIR, 'J_regressor_h36m.npy'))).float()


    """ Data """
    t_total = 90
    overlap = 10
    out_dir = f'./output/{target_dataset}_test_output'
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if target_dataset == '3dpw':
        data_path = f'data/fthe_db/{target_dataset}_{set}_db.pt'  #
    elif target_dataset == 'mpii3d':
        set = 'val'
        data_path = f'data/fthe_db/{target_dataset}_{set}_db.pt'  #
    else:
        print("Wrong target dataset! Exiting...")
        import sys; sys.exit()

    print(f"Load data from {data_path}")
    dataset_data = joblib.load(data_path)
    full_res = defaultdict(list)

    vid_name_list = dataset_data['vid_name']
    unique_names = np.unique(vid_name_list)
    data_keyed = {}

    # make dictionary with video seqeunce names
    for u_n in unique_names:
        if (target_action != '') and (not target_action in u_n):
            continue
        indexes = vid_name_list == u_n
        if 'valid' in dataset_data:
            valids = dataset_data['valid'][indexes].astype(bool)
        else:
            valids = np.ones(dataset_data['features'][indexes].shape[0]).astype(bool)
        # import pdb; pdb.set_trace()
        # valids[:] = 1
        data_keyed[u_n] = {
            'features': dataset_data['features'][indexes][valids],
            'joints3D': dataset_data['joints3D'][indexes][valids],
            'vid_name': dataset_data['vid_name'][indexes][valids],
            'imgname': dataset_data['img_name'][indexes][valids],
            'bbox': dataset_data['bbox'][indexes][valids],
        }
        if 'mpii3d' in data_path:
            data_keyed[u_n]['pose'] = np.zeros((len(valids), 72))
            data_keyed[u_n]['shape'] = np.zeros((len(valids), 10))
            data_keyed[u_n]['valid_i'] = dataset_data['valid_i'][indexes][valids]
            J_regressor = None
        else:
            data_keyed[u_n]['pose'] = dataset_data['pose'][indexes][valids]
            data_keyed[u_n]['shape'] = dataset_data['shape'][indexes][valids]
    dataset_data = data_keyed
    result = {}
    """ Run evaluation """
    model.eval()
    with torch.no_grad():
        tot_num_pose = 0
        pbar = tqdm(dataset_data.keys())
        for seq_name in pbar:
            curr_feats = dataset_data[seq_name]['features']
            res_save = {}
            curr_feat = torch.tensor(curr_feats).to(device)
            num_frames = curr_feat.shape[0]

            if num_frames < t_total:
                if num_frames < t_total:
                    print(f"Video < {t_total} frames")
                print(f"video too short, padding..... {num_frames}")
                curr_feat = torch.from_numpy(np.repeat(curr_feats, t_total//num_frames + 1, axis = 0)[:t_total].copy()).to(device)
                chunk_idxes = np.array(list(range(0, t_total)))[None, ]
                chunck_selects = [(0, num_frames)]
            else:
                chunk_idxes, chunck_selects = get_chunk_with_overlap(num_frames, window_size=t_total, overlap=overlap)

            vid_names = dataset_data[seq_name]['vid_name']

            pred_j3ds, pred_verts, pred_rotmats, pred_thetas = [], [], [], []
            for curr_idx in range(len(chunk_idxes)):
                chunk_idx = chunk_idxes[curr_idx]
                cl = chunck_selects[curr_idx]
                input_feat = curr_feat[None, chunk_idx, :]

                preds = model(input_feat, J_regressor=J_regressor)
                n_kp = preds[-1]['kp_3d'].shape[-2]
                pred_theta = preds[-1]['theta'][0, cl[0]:cl[1],3:75].cpu().numpy()
                pred_j3d = preds[-1]['kp_3d'][0, cl[0]:cl[1]].cpu().numpy()
                pred_vert = preds[-1]['verts'][0, cl[0]:cl[1]].cpu().numpy()
                pred_rotmat = preds[-1]['rotmat'][0, cl[0]:cl[1]].cpu().numpy()



                pred_j3ds.append(pred_j3d)
                pred_verts.append(pred_vert)
                pred_rotmats.append(pred_rotmat)
                pred_thetas.append(pred_theta)

            # temporal smoothing post-processing following MEVA (https://github.com/ZhengyiLuo/MEVA)
            if avg_filter:
                # slerp avg filter
                pred_thetas = np.vstack(pred_thetas).astype(np.float32)
                pred_rotmats = np.vstack(pred_rotmats)
                pred_rotmats = smooth_pose_mat(np.array(pred_rotmats), ratio=0.3).astype(np.float32)

                smpl = SMPL(model_path=SMPL_MODEL_DIR)
                smpl_output = smpl(
                    betas=torch.from_numpy(pred_thetas[:, 75:]),
                    body_pose=torch.from_numpy(pred_rotmats[:, 1:]),
                    global_orient=torch.from_numpy(pred_rotmats[:, 0:1]),
                    pose2rot=False,
                )
                filtered_pred_verts = smpl_output.vertices
                # for render
                pred_vertes = filtered_pred_verts
                J_regressor_batch = J_regressor[None, :].expand(filtered_pred_verts.shape[0], -1, -1)
                pred_joints = torch.matmul(J_regressor_batch, filtered_pred_verts)
                pred_j3ds = pred_joints[:, H36M_TO_J14, :].detach().cpu().numpy()
            else:
                try:
                    pred_j3ds = np.vstack(pred_j3ds)
                except:
                    import pdb; pdb.set_trace()
            target_j3ds = dataset_data[seq_name]['joints3D']
            pred_verts = np.vstack(pred_verts)
            dummy_cam = np.repeat(np.array([[1., 0., 0.]]), len(target_j3ds), axis=0)
            target_theta = np.concatenate([dummy_cam, dataset_data[seq_name]['pose'], dataset_data[seq_name]['shape']], axis=1).astype(np.float32)
            target_j3ds, target_theta = target_j3ds[:len(pred_j3ds)], target_theta[:len(pred_j3ds)]

            """ Rendering """
            if render:
                num_frames_to_render = 1000
                imgname = dataset_data[seq_name]['imgname']
                bbox = dataset_data[seq_name]['bbox']
                pred_cam = np.vstack(pred_thetas).astype(np.float32)[:, :3]
                img = cv2.imread(imgname[0])
                orig_height, orig_width = img.shape[:2]
                renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=False)

                if render_plain:
                    save_seq_name = f'{seq_name}_plain'
                elif only_img:
                    save_seq_name = f'{seq_name}_input'
                else:
                    save_seq_name = seq_name
                save_seq_name = "renderer" + save_seq_name + '_' + str(render_frame_start)

                count = 0
                for ii in tqdm(range(len(imgname))):
                    frame_i = int(imgname[ii].split('_')[-1][:-4])
                    if (frame_i < render_frame_start) or (frame_i > render_frame_start+num_frames_to_render):
                        continue
                    count += 1

                    Path(osp.join(out_dir, save_seq_name)).mkdir(parents=True, exist_ok=True)

                    bbox_ii = bbox[0:1].copy() if render_plain else bbox[ii:ii + 1]
                    bbox_ii[:, 2:] = bbox_ii[:, 2:] * 1.2

                    img_path = imgname[ii]
                    img = cv2.imread(img_path)
                    cam = np.array([[1, 0, 0]]) if render_plain else pred_cam[ii:ii + 1]
                    orig_cam = convert_crop_cam_to_orig_img(
                        cam=cam,
                        bbox=bbox_ii,
                        img_width=orig_width,
                        img_height=orig_height
                    )

                    # if not only_img:
                    #     # try:
                    #     # if render_plain:
                    #     #     img[:] = 0
                    #     img = renderer.render(
                    #         img,
                    #         pred_verts[ii],
                    #         cam=orig_cam[0],
                    #         color=[1.0, 1.0, 0.9],
                    #         mesh_filename=None,
                    #     )
                        # except:
                        #     print("Error on rendering! Exiting...")
                        #     import sys; sys.exit()

                    # resize image to save storage
                    h, w = img.shape[:2]
                    new_h, new_w = int(h/2), int(w/2)
                    new_h, new_w = new_h if new_h % 2 == 0 else new_h-1, new_w if new_w % 2 == 0 else new_w-1  # for ffmpeg
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    new_height, new_width = img.shape[:2]


                    cv2.imwrite(osp.join(out_dir, save_seq_name, f'{count:06d}.png'), img)

                save_path = osp.join(out_dir, 'video', save_seq_name + ".mp4")
                Path(osp.join(out_dir, 'video')).mkdir(parents=True, exist_ok=True)
                print(f"Saving result video to {osp.abspath(save_path)}")
                images_to_video(img_folder=osp.join(out_dir, save_seq_name), output_vid_file=save_path)

                shutil.rmtree(osp.join(out_dir, save_seq_name))

            if 'mpii3d' in data_path:
                target_j3ds = convert_kps(target_j3ds, src='spin', dst='mpii3d_test')
                pred_j3ds = convert_kps(pred_j3ds, src='spin', dst='mpii3d_test')

                valid_map = dataset_data[seq_name]['valid_i'][:,0].nonzero()[0]
                if valid_map.size == 0:
                    print("No valid frames. Continue")  # 'subj6_seg0'
                    continue
                while True:
                    if valid_map[-1] >= len(pred_j3ds):
                        valid_map = valid_map[:-1]
                    else:
                        break

            elif target_j3ds.shape[1] == 49:
                target_j3ds = convert_kps(target_j3ds, src='spin', dst='common')
                valid_map = np.arange(len(target_j3ds))
            else:
                valid_map = np.arange(len(target_j3ds))

            pred_j3ds = torch.from_numpy(pred_j3ds).float()
            target_j3ds = torch.from_numpy(target_j3ds).float()

            num_eval_pose = len(valid_map)
            print(f"Evaluating on {num_eval_pose} data (number of poses) in {seq_name}...")
            tot_num_pose += num_eval_pose

            if 'mpii3d' in data_path:
                pred_pelvis = pred_j3ds[:, [-3], :]
                target_pelvis = target_j3ds[:, [-3], :]
            else:
                pred_pelvis = (pred_j3ds[:, [2], :] + pred_j3ds[:, [3], :]) / 2.0
                target_pelvis = (target_j3ds[:, [2], :] + target_j3ds[:, [3], :]) / 2.0

            pred_j3ds -= pred_pelvis
            target_j3ds -= target_pelvis

            m2mm = 1000
            # per-frame accuracy
            mpvpe = compute_error_verts(target_theta=target_theta, pred_verts=pred_verts) * m2mm
            mpjpe = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).cpu().numpy()[valid_map]
            mpjpe = mpjpe.mean(axis=-1) * m2mm
            S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
            mpjpe_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).cpu().numpy()[valid_map]
            mpjpe_pa = mpjpe_pa.mean(axis=-1) * m2mm
            # acceleration error
            if plot:
                plot_accel(pred_j3ds, joints_gt=target_j3ds, out_dir=out_dir, name=target_action)
            accel = compute_accel(pred_j3ds) * m2mm
            accel_err = np.zeros((len(pred_j3ds,)))
            accel_err[1:-1] = compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds) * m2mm
            # exclude 0 from accel error calculation



            if valid_map[0] == 0:
                valid_map = valid_map[1:]
            # if valid_map[-1] == len(accel_err)-1:
            #     valid_map = valid_map[:-1]

            full_res['mpjpe'].append(mpjpe)
            full_res['mpjpe_pa'].append(mpjpe_pa)
            full_res['accel'].append(accel)
            if target_dataset == '3dpw':
                full_res['mpvpe'].append(mpvpe)
            pbar.set_description(f"{np.mean(mpjpe_pa):.3f}")
            # result.update({seq_name: str(np.mean(accel))+' '+str(np.mean(mpjpe_pa))})

        print(f"\nEvaluated total {tot_num_pose} poses")
        full_res.pop(0, None)
        full_res = {k: np.mean(np.concatenate(v)) for k, v in full_res.items()}
        print(full_res)
        # print(result)
