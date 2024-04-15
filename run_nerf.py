# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from icecream import ic
from test_without_manager import *
import argparse
import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
from segmentation_utils import Segmenter
from PIL import Image

def run_one_video(mani_reader, out_folder = './tmp', debug_level=3, stride = 5):
  # set_seed(0)
  os.system(f'rm -rf {out_folder} && mkdir -p {out_folder}')
  cfg_bundletrack = yaml.load(open(f"{code_dir}/BundleTrack/config_ho3d.yml",'r'))
  cfg_bundletrack['SPDLOG'] = int(debug_level)
  cfg_bundletrack['depth_processing']["zfar"] = 1
  cfg_bundletrack['depth_processing']["percentile"] = 95
  cfg_bundletrack['erode_mask'] = 3
  cfg_bundletrack['debug_dir'] = out_folder+'/'
  cfg_bundletrack['bundle']['max_BA_frames'] = 10
  cfg_bundletrack['bundle']['max_optimized_feature_loss'] = 0.03
  cfg_bundletrack['feature_corres']['max_dist_neighbor'] = 0.02
  cfg_bundletrack['feature_corres']['max_normal_neighbor'] = 30
  cfg_bundletrack['feature_corres']['max_dist_no_neighbor'] = 0.01
  cfg_bundletrack['feature_corres']['max_normal_no_neighbor'] = 20
  cfg_bundletrack['feature_corres']['map_points'] = True
  cfg_bundletrack['feature_corres']['resize'] = 400
  cfg_bundletrack['feature_corres']['rematch_after_nerf'] = True
  cfg_bundletrack['keyframe']['min_rot'] = 5
  cfg_bundletrack['ransac']['inlier_dist'] = 0.01
  cfg_bundletrack['ransac']['inlier_normal_angle'] = 20
  cfg_bundletrack['ransac']['max_trans_neighbor'] = 0.02
  cfg_bundletrack['ransac']['max_rot_deg_neighbor'] = 30
  cfg_bundletrack['ransac']['max_trans_no_neighbor'] = 0.01
  cfg_bundletrack['ransac']['max_rot_no_neighbor'] = 10
  cfg_bundletrack['p2p']['max_dist'] = 0.02
  cfg_bundletrack['p2p']['max_normal_angle'] = 45
  cfg_track_dir = f'{out_folder}/config_bundletrack.yml'
  yaml.dump(cfg_bundletrack, open(cfg_track_dir,'w'))

  cfg_nerf = yaml.load(open(f"{code_dir}/config.yml",'r'))
  cfg_nerf['continual'] = True
  cfg_nerf['trunc_start'] = 0.01
  cfg_nerf['trunc'] = 0.01
  cfg_nerf['mesh_resolution'] = 0.005
  cfg_nerf['down_scale_ratio'] = 1
  cfg_nerf['fs_sdf'] = 0.1
  cfg_nerf['far'] = cfg_bundletrack['depth_processing']["zfar"]
  cfg_nerf['datadir'] = f"{cfg_bundletrack['debug_dir']}/nerf_with_bundletrack_online"
  cfg_nerf['notes'] = ''
  cfg_nerf['expname'] = 'nerf_with_bundletrack_online'
  cfg_nerf['save_dir'] = cfg_nerf['datadir']
  cfg_nerf_dir = f'{out_folder}/config_nerf.yml'
  yaml.dump(cfg_nerf, open(cfg_nerf_dir,'w'))
  tracker = BundleSdf(cfg_track_dir=cfg_track_dir, cfg_nerf_dir=cfg_nerf_dir, start_nerf_keyframes=100)
  

  for i in range(0,len(mani_reader),stride):
    color = mani_reader.get_rgb(i)
    depth = mani_reader.get_depth(i)
    mask = mani_reader.get_mask(i)
    id_str = str(i)
    pose_in_model = np.eye(4)
    if cfg_bundletrack['erode_mask']>0:
      kernel = np.ones((cfg_bundletrack['erode_mask'], cfg_bundletrack['erode_mask']), np.uint8)
      mask = cv2.erode(mask.astype(np.uint8), kernel)
    K = mani_reader.K.copy()
    tracker.run(color, depth, K, id_str, mask=mask, occ_mask=None, pose_in_model=pose_in_model)

  tracker.on_finish()
  return run_one_video_global_nerf(out_folder, mani_reader)

def run_one_video_global_nerf(out_folder='/home/bowen/debug/bundlesdf_scan_coffee_415', mani_reader=None):
  # set_seed(0)

  out_folder += '/'   #!NOTE there has to be a / in the end

  cfg_bundletrack = yaml.load(open(f"{out_folder}/config_bundletrack.yml",'r'))
  cfg_bundletrack['debug_dir'] = out_folder
  cfg_track_dir = f"{out_folder}/config_bundletrack.yml"
  yaml.dump(cfg_bundletrack, open(cfg_track_dir,'w'))

  cfg_nerf = yaml.load(open(f"{out_folder}/config_nerf.yml",'r'))
  cfg_nerf['n_step'] = 500
  cfg_nerf['N_samples'] = 64
  cfg_nerf['N_samples_around_depth'] = 256
  cfg_nerf['first_frame_weight'] = 1
  cfg_nerf['down_scale_ratio'] = 1
  cfg_nerf['finest_res'] = 256
  cfg_nerf['num_levels'] = 16
  cfg_nerf['mesh_resolution'] = 0.002
  cfg_nerf['n_train_image'] = 500
  cfg_nerf['fs_sdf'] = 0.1
  cfg_nerf['frame_features'] = 2
  cfg_nerf['rgb_weight'] = 100

  cfg_nerf['i_img'] = np.inf
  cfg_nerf['i_mesh'] = cfg_nerf['i_img']
  cfg_nerf['i_nerf_normals'] = cfg_nerf['i_img']
  cfg_nerf['i_save_ray'] = cfg_nerf['i_img']

  cfg_nerf['datadir'] = f"{out_folder}/nerf_with_bundletrack_online"
  cfg_nerf['save_dir'] = copy.deepcopy(cfg_nerf['datadir'])

  os.makedirs(cfg_nerf['datadir'],exist_ok=True)

  cfg_nerf_dir = f"{cfg_nerf['datadir']}/config.yml"
  yaml.dump(cfg_nerf, open(cfg_nerf_dir,'w'))

  tracker = BundleSdf(cfg_track_dir=cfg_track_dir, cfg_nerf_dir=cfg_nerf_dir, start_nerf_keyframes=5)
  tracker.cfg_nerf = cfg_nerf
  
  pcd = tracker.run_global_nerf(reader=mani_reader, get_texture=True, tex_res=512)
  tracker.on_finish()


  # print(f"Done")
  return pcd

class ManiImageReader:
  def __init__(self, rgbs, depths, masks, K, shorter_side=480, unit=1e3) -> None:
    '''
      rbgs: n*w*h*3 rgb
      depths: unit mm
    '''
    self.W, self.H = rgbs.shape[1:3]
    self.downscale = shorter_side/min(self.H, self.W)
    self.H = int(self.H*self.downscale)
    self.W = int(self.W*self.downscale)
    self.rgbs = rgbs
    self.depths = depths/unit
    self.masks = masks
    self.K = K
    self.K[:2] *= self.downscale
  def get_rgb(self, idx):
    img = self.rgbs[idx]
    return cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
  def get_mask(self, idx):
    img = self.masks[idx]
    if len(img.shape)==3:
      img = (img.sum(axis=-1)>0).astype(np.uint8)
    return cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
  def get_depth(self, idx):
    img = self.depths[idx]
    return cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
  def __len__(self):
    return self.masks.shape[0]

if __name__=="__main__":
  set_seed(0)
  parser = argparse.ArgumentParser()
  parser.add_argument('--video_dir', type=str, default="/home/bowen/debug/2022-11-18-15-10-24_milk/")
  parser.add_argument('--debug_level', type=int, default=2, help='higher means more logging')
  args = parser.parse_args()
  rgbs = []
  depth = []
  masks = []
  files = sorted(os.listdir(os.path.join(args.video_dir,'rgb')), key=lambda x: int(re.findall(r'\d+', x)[0]))
  for file in files:
    if file.endswith('.png'):
      rgbs.append(np.array(Image.open(os.path.join(args.video_dir, 'rgb', file))))

  files = sorted(os.listdir(os.path.join(args.video_dir,'masks')), key=lambda x: int(re.findall(r'\d+', x)[0]))
  for file in files:
    if file.endswith('.png'):
      masks.append(np.array(Image.open(os.path.join(args.video_dir, 'masks', file))))

  files = sorted(os.listdir(os.path.join(args.video_dir,'depth')), key=lambda x: int(re.findall(r'\d+', x)[0]))
  for file in files:
    if file.endswith('.png'):
      depth.append(np.array(Image.open(os.path.join(args.video_dir, 'depth', file))))
  rgbs = np.stack(rgbs, axis=0)[:,:,:,::-1]
  masks = np.stack(masks, axis=0)
  depth = np.stack(depth, axis=0)
  K = np.loadtxt(os.path.join(args.video_dir, 'cam_K.txt')).reshape(3,3)
  manireader = ManiImageReader(rgbs, depth, masks, K)
  run_one_video(manireader)

