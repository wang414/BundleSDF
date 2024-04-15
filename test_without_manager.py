# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
from nerf_runner import *
from tool import *
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/BundleTrack/build')
import my_cpp
from gui import *
from BundleTrack.scripts.data_reader import *
from Utils import *
from loftr_wrapper import LoftrRunner


class BundleSdf:
  def __init__(self, cfg_track_dir=f"{code_dir}/config_ho3d.yml", cfg_nerf_dir=f'{code_dir}/config.yml', start_nerf_keyframes=10, translation=None, sc_factor=None):
    with open(cfg_track_dir,'r') as ff:
      self.cfg_track = yaml.load(ff)
    self.debug_dir = self.cfg_track["debug_dir"]
    self.SPDLOG = self.cfg_track["SPDLOG"]
    self.start_nerf_keyframes = start_nerf_keyframes
    self.translation = None
    self.sc_factor = None
    if sc_factor is not None:
      self.translation = translation
      self.sc_factor = sc_factor
    self.use_gui = False
    code_dir = os.path.dirname(os.path.realpath(__file__))
    with open(cfg_nerf_dir,'r') as ff:
      self.cfg_nerf = yaml.load(ff)
    self.cfg_nerf['notes'] = ''
    self.cfg_nerf['bounding_box'] = np.array(self.cfg_nerf['bounding_box']).reshape(2,3)
    yml = my_cpp.YamlLoadFile(cfg_track_dir)
    self.bundler = my_cpp.Bundler(yml)
    self.loftr = LoftrRunner()
    self.cnt = -1
    self.K = None
    self.mesh = None


  def on_finish(self):
    return


  def make_frame(self, color, depth, K, id_str, mask=None, occ_mask=None, pose_in_model=np.eye(4)):
    H,W = color.shape[:2]
    roi = [0,W-1,0,H-1]
    frame = my_cpp.Frame(color,depth,roi,pose_in_model,self.cnt,id_str,K,self.bundler.yml)
    if mask is not None:
      frame._fg_mask = my_cpp.cvMat(mask)
    if occ_mask is not None:
      frame._occ_mask = my_cpp.cvMat(occ_mask)
    return frame


  def find_corres(self, frame_pairs):
    # logging.info(f"frame_pairs: {len(frame_pairs)}")
    is_match_ref = len(frame_pairs)==1 and frame_pairs[0][0]._ref_frame_id==frame_pairs[0][1]._id and self.bundler._newframe==frame_pairs[0][0]

    imgs, tfs, query_pairs = self.bundler._fm.getProcessedImagePairs(frame_pairs)
    imgs = np.array([np.array(img) for img in imgs])

    if len(query_pairs)==0:
      return

    corres = self.loftr.predict(rgbAs=imgs[::2], rgbBs=imgs[1::2])
    for i_pair in range(len(query_pairs)):
      cur_corres = corres[i_pair][:,:4]
      tfA = np.array(tfs[i_pair*2])
      tfB = np.array(tfs[i_pair*2+1])
      cur_corres[:,:2] = transform_pts(cur_corres[:,:2], np.linalg.inv(tfA))
      cur_corres[:,2:4] = transform_pts(cur_corres[:,2:4], np.linalg.inv(tfB))
      self.bundler._fm._raw_matches[query_pairs[i_pair]] = cur_corres.round().astype(np.uint16)

    min_match_with_ref = self.cfg_track["feature_corres"]["min_match_with_ref"]

    if is_match_ref and len(self.bundler._fm._raw_matches[frame_pairs[0]])<min_match_with_ref:
      self.bundler._fm._raw_matches[frame_pairs[0]] = []
      self.bundler._newframe._status = my_cpp.Frame.FAIL
    #   logging.info(f'frame {self.bundler._newframe._id_str} mark FAIL, due to no matching')
      return

    self.bundler._fm.rawMatchesToCorres(query_pairs)

    for pair in query_pairs:
      self.bundler._fm.vizCorresBetween(pair[0], pair[1], 'before_ransac')

    self.bundler._fm.runRansacMultiPairGPU(query_pairs)

    for pair in query_pairs:
      self.bundler._fm.vizCorresBetween(pair[0], pair[1], 'after_ransac')



  def process_new_frame(self, frame):
    # logging.info(f"process frame {frame._id_str}")

    self.bundler._newframe = frame
    os.makedirs(self.debug_dir, exist_ok=True)

    if frame._id>0:
      ref_frame = self.bundler._frames[list(self.bundler._frames.keys())[-1]]
      frame._ref_frame_id = ref_frame._id
      frame._pose_in_model = ref_frame._pose_in_model
    else:
      self.bundler._firstframe = frame

    frame.invalidatePixelsByMask(frame._fg_mask)
    if frame._id==0 and np.abs(np.array(frame._pose_in_model)-np.eye(4)).max()<=1e-4:
      frame.setNewInitCoordinate()


    n_fg = (np.array(frame._fg_mask)>0).sum()
    if n_fg<100:
    #   logging.info(f"Frame {frame._id_str} cloud is empty, marked FAIL, roi={n_fg}")
      frame._status = my_cpp.Frame.FAIL;
      self.bundler.forgetFrame(frame)
      return

    if self.cfg_track["depth_processing"]["denoise_cloud"]:
      frame.pointCloudDenoise()

    n_valid = frame.countValidPoints()
    n_valid_first = self.bundler._firstframe.countValidPoints()
    if n_valid<n_valid_first/40.0:
    #   logging.info(f"frame _cloud_down points#: {n_valid} too small compared to first frame points# {n_valid_first}, mark as FAIL")
      frame._status = my_cpp.Frame.FAIL
      self.bundler.forgetFrame(frame)
      return

    if frame._id==0:
      self.bundler.checkAndAddKeyframe(frame)   # First frame is always keyframe
      self.bundler._frames[frame._id] = frame
      return

    min_match_with_ref = self.cfg_track["feature_corres"]["min_match_with_ref"]

    self.find_corres([(frame, ref_frame)])
    matches = self.bundler._fm._matches[(frame, ref_frame)]

    if frame._status==my_cpp.Frame.FAIL:
    #   logging.info(f"find corres fail, mark {frame._id_str} as FAIL")
      self.bundler.forgetFrame(frame)
      return

    matches = self.bundler._fm._matches[(frame, ref_frame)]
    if len(matches)<min_match_with_ref:
      visibles = []
      for kf in self.bundler._keyframes:
        visible = my_cpp.computeCovisibility(frame, kf)
        visibles.append(visible)
      visibles = np.array(visibles)
      ids = np.argsort(visibles)[::-1]
      found = False
      # pdb.set_trace()
      for id in ids:
        kf = self.bundler._keyframes[id]
        # logging.info(f"trying new ref frame {kf._id_str}")
        ref_frame = kf
        frame._ref_frame_id = kf._id
        frame._pose_in_model = kf._pose_in_model
        self.find_corres([(frame, ref_frame)])

        # self.bundler._fm.findCorres(frame, ref_frame)

        if len(self.bundler._fm._matches[(frame,kf)])>=min_match_with_ref:
        #   logging.info(f"re-choose new ref frame to {kf._id_str}")
          found = True
          break

      if not found:
        frame._status = my_cpp.Frame.FAIL
        # logging.info(f"frame {frame._id_str} has not suitable ref_frame, mark as FAIL")
        self.bundler.forgetFrame(frame)
        return

    # logging.info(f"frame {frame._id_str} pose update before\n{frame._pose_in_model.round(3)}")
    offset = self.bundler._fm.procrustesByCorrespondence(frame, ref_frame)
    frame._pose_in_model = offset@frame._pose_in_model
    # logging.info(f"frame {frame._id_str} pose update after\n{frame._pose_in_model.round(3)}")

    window_size = self.cfg_track["bundle"]["window_size"]
    if len(self.bundler._frames)-len(self.bundler._keyframes)>window_size:
      for k in self.bundler._frames:
        f = self.bundler._frames[k]
        isforget = self.bundler.forgetFrame(f)
        if isforget:
        #   logging.info(f"exceed window size, forget frame {f._id_str}")
          break

    self.bundler._frames[frame._id] = frame

    self.bundler.selectKeyFramesForBA()

    local_frames = self.bundler._local_frames

    pairs = self.bundler.getFeatureMatchPairs(self.bundler._local_frames)
    self.find_corres(pairs)
    if frame._status==my_cpp.Frame.FAIL:
      self.bundler.forgetFrame(frame)
      return

    find_matches = False
    self.bundler.optimizeGPU(local_frames, find_matches)

    if frame._status==my_cpp.Frame.FAIL:
      self.bundler.forgetFrame(frame)
      return

    self.bundler.checkAndAddKeyframe(frame)



  def run(self, color, depth, K, id_str, mask=None, occ_mask=None, pose_in_model=np.eye(4)):
    self.cnt += 1

    if self.K is None:
      self.K = K

    H,W = color.shape[:2]
    percentile = self.cfg_track['depth_processing']["percentile"]
    # if percentile<100:   # Denoise
    #   logging.info("percentile denoise start")
    #   valid = (depth>=0.1) & (mask>0)
    #   thres = np.percentile(depth[valid], percentile)
    #   depth[depth>=thres] = 0
    #   logging.info("percentile denoise done")
    # print(color.sum(), depth.sum(), mask.sum(), mask.max())
    # print(color, depth, K, id_str, mask, occ_mask, pose_in_model)
    
    # exit()
    frame = self.make_frame(color, depth, K, id_str, mask, occ_mask, pose_in_model)
    os.makedirs(f"{self.debug_dir}/{frame._id_str}", exist_ok=True)

    # logging.info(f"processNewFrame start {frame._id_str}")
    # self.bundler.processNewFrame(frame)
    self.process_new_frame(frame)
    # logging.info(f"processNewFrame done {frame._id_str}")


    rematch_after_nerf = self.cfg_track["feature_corres"]["rematch_after_nerf"]
    # logging.info(f"rematch_after_nerf: {rematch_after_nerf}")
    frames_large_update = []

    if rematch_after_nerf:
      if len(frames_large_update)>0:
        # logging.info(f"before matches keys: {len(self.bundler._fm._matches)}")
        ks = list(self.bundler._fm._matches.keys())
        for k in ks:
          if k[0] in frames_large_update or k[1] in frames_large_update:
            del self.bundler._fm._matches[k]
        #     logging.info(f"Delete match between {k[0]._id_str} and {k[1]._id_str}")
        # logging.info(f"after matches keys: {len(self.bundler._fm._matches)}")

    self.bundler.saveNewframeResult()
    if self.SPDLOG>=2 and occ_mask is not None:
      os.makedirs(f'{self.debug_dir}/occ_mask/', exist_ok=True)
      cv2.imwrite(f'{self.debug_dir}/occ_mask/{frame._id_str}.png', occ_mask)

    if self.use_gui:
      ob_in_cam = np.linalg.inv(frame._pose_in_model)
      with self.gui_lock:
        self.gui_dict['color'] = color[...,::-1]
        self.gui_dict['mask'] = mask
        self.gui_dict['ob_in_cam'] = ob_in_cam
        self.gui_dict['id_str'] = frame._id_str
        self.gui_dict['K'] = self.K
        self.gui_dict['n_keyframe'] = len(self.bundler._keyframes)



  def run_global_nerf(self, reader=None, get_texture=False, tex_res=1024):
    '''
    @reader: data reader, sometimes we want to use the full resolution raw image
    '''
    self.K = np.loadtxt(f'{self.debug_dir}/cam_K.txt').reshape(3,3)
    # sorted(os.listdir(os.path.join(args.video_dir,'rgb')), key=lambda x: int(re.findall(r'\d+', x)[0]))
    tmp = sorted(glob.glob(f"{self.debug_dir}/ob_in_cam/*"), key=lambda x: int(re.findall(r'\d+', x)[0]))
    last_stamp = os.path.basename(tmp[-1]).replace('.txt','')
    # logging.info(f'last_stamp {last_stamp}')
    keyframes = yaml.load(open(f'{self.debug_dir}/{last_stamp}/keyframes.yml','r'))
    # logging.info(f"keyframes#: {len(keyframes)}")
    keys = list(keyframes.keys())

    frame_ids = []
    for k in keys:
      frame_ids.append(k.replace('keyframe_',''))

    cam_in_obs = []
    for k in keys:
      cam_in_ob = np.array(keyframes[k]['cam_in_ob']).reshape(4,4)
      cam_in_obs.append(cam_in_ob)
    cam_in_obs = np.array(cam_in_obs)

    out_dir = f"{self.debug_dir}/final/nerf"
    os.system(f"rm -rf {out_dir} && mkdir -p {out_dir}")
    os.system(f'rm -rf {self.debug_dir}/final/used_rgbs/ && mkdir -p {self.debug_dir}/final/used_rgbs/')

    rgbs = []
    depths = []
    normal_maps = []
    masks = []
    occ_masks = []
    for frame_id in frame_ids:
      self.K = reader.K.copy()
      id = int(frame_id)
      rgbs.append(reader.get_rgb(id))
      depths.append(reader.get_depth(id))
      masks.append(reader.get_mask(id))


    glcam_in_obs = cam_in_obs@glcam_in_cvcam

    self.cfg_nerf['sc_factor'] = None
    self.cfg_nerf['translation'] = None

    ######### Reuse normalization
    files = sorted(glob.glob(f"{self.debug_dir}/**/nerf/config.yml", recursive=True))
    if len(files)>0:
      tmp = yaml.load(open(files[-1],'r'))
      self.cfg_nerf['sc_factor'] = float(tmp['sc_factor'])
      self.cfg_nerf['translation'] = np.array(tmp['translation'])

    sc_factor,translation,pcd_real_scale, pcd_normalized = compute_scene_bounds(None,glcam_in_obs,self.K,use_mask=True,base_dir=self.cfg_nerf['save_dir'],rgbs=np.array(rgbs),depths=np.array(depths),masks=np.array(masks), cluster=True, eps=0.01, min_samples=5, sc_factor=self.cfg_nerf['sc_factor'], translation_cvcam=self.cfg_nerf['translation'])
    pcd = np.asarray(pcd_normalized.points)
    # color = np.asarray(pcd_normalized.colors)
    # pcd_normalized.colors = o3d.cuda.pybind.utility.Vector3dVector(color[:,::-1])
    # o3d.io.write_point_cloud(f"{out_dir}/point_cloud.ply", pcd_normalized)
    return pcd
    # self.cfg_nerf['sc_factor'] = float(sc_factor)
    # self.cfg_nerf['translation'] = translation

    # if normal_maps is not None and len(normal_maps)>0:
    #   normal_maps = np.array(normal_maps)
    # else:
    #   normal_maps = None

    # rgbs_raw = np.array(rgbs).copy()
    # rgbs,depths,masks,normal_maps,poses = preprocess_data(np.array(rgbs),depths=np.array(depths),masks=np.array(masks),normal_maps=normal_maps,poses=glcam_in_obs,sc_factor=self.cfg_nerf['sc_factor'],translation=self.cfg_nerf['translation'])

    # self.cfg_nerf['sampled_frame_ids'] = np.arange(len(rgbs))

    # np.savetxt(f"{self.cfg_nerf['save_dir']}/trainval_poses.txt",glcam_in_obs.reshape(-1,4))

    # if len(occ_masks)>0:
    #   occ_masks = np.array(occ_masks)
    # else:
    #   occ_masks = None

    # nerf = NerfRunner(self.cfg_nerf,rgbs,depths=depths,masks=masks,normal_maps=normal_maps,occ_masks=occ_masks,poses=poses,K=self.K,build_octree_pcd=pcd_normalized)
    # exit()
    # print("Start training")
    # nerf.train()
    # optimized_cvcam_in_obs,offset = get_optimized_poses_in_real_world(poses,nerf.models['pose_array'],self.cfg_nerf['sc_factor'],self.cfg_nerf['translation'])

    # ####### Log
    # # os.system(f"cp -r {self.cfg_nerf['save_dir']}/image_step_*.png  {out_dir}/")
    # with open(f"{out_dir}/config.yml",'w') as ff:
    #   tmp = copy.deepcopy(self.cfg_nerf)
    #   for k in tmp.keys():
    #     if isinstance(tmp[k],np.ndarray):
    #       tmp[k] = tmp[k].tolist()
    #   yaml.dump(tmp,ff)
    # shutil.copy(f"{out_dir}/config.yml",f"{self.cfg_nerf['save_dir']}/")
    # os.system(f"mv {self.cfg_nerf['save_dir']}/*  {out_dir}/ && rm -rf {out_dir}/step_*_mesh_real_world.obj {out_dir}/*frame*ray*.ply")

    # torch.cuda.empty_cache()

    # np.savetxt(f"{self.debug_dir}/{frame_id}/poses_after_nerf.txt",np.array(optimized_cvcam_in_obs).reshape(-1,4))

    # # mesh_files = sorted(glob.glob(f"{self.debug_dir}/final/nerf/step_*_mesh_normalized_space.obj"))
    # # mesh = trimesh.load(mesh_files[-1])

    # mesh,sigma,query_pts = nerf.extract_mesh(voxel_size=self.cfg_nerf['mesh_resolution'],isolevel=0, return_sigma=True)
    # mesh.export(f'{self.debug_dir}/try.ply')
    # mesh.merge_vertices()
    # ms = trimesh_split(mesh, min_edge=100)
    # largest_size = 0
    # largest = None
    # for m in ms:
    #   # mean = m.vertices.mean(axis=0)
    #   # if np.linalg.norm(mean)>=0.1*nerf.cfg['sc_factor']:
    #   #   continue
    #   if m.vertices.shape[0]>largest_size:
    #     largest_size = m.vertices.shape[0]
    #     largest = m
    # mesh = largest
    # mesh.export(f'{self.debug_dir}/mesh_cleaned.obj')

    # if get_texture:
    #   mesh = nerf.mesh_texture_from_train_images(mesh, rgbs_raw=rgbs_raw, train_texture=False, tex_res=tex_res)

    # mesh = mesh_to_real_world(mesh, pose_offset=offset, translation=self.cfg_nerf['translation'], sc_factor=self.cfg_nerf['sc_factor'])
    # mesh.export(f'{self.debug_dir}/textured_mesh.obj')





if __name__=="__main__":
  set_seed(0)
  torch.set_default_tensor_type('torch.cuda.FloatTensor')

  cfg_nerf = yaml.load(open(f"{code_dir}/BundleTrack/config_ho3d.yml",'r'))
  cfg_nerf['data_dir'] = '/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/HO3D_v3/evaluation/MPM13'
  cfg_nerf['SPDLOG'] = 1

  cfg_track_dir = '/tmp/config.yml'
  yaml.dump(cfg_nerf, open(cfg_track_dir,'w'))
  tracker = BundleSdf(cfg_track_dir=cfg_track_dir)
  reader = Ho3dReader(tracker.bundler.yml["data_dir"].Scalar())

  os.system(f"rm -rf {tracker.debug_dir} && mkdir -p {tracker.debug_dir}")

  for i,color_file in enumerate(reader.color_files):
    color = cv2.imread(color_file)
    depth = reader.get_depth(i)
    id_str = reader.id_strs[i]
    occ_mask = reader.get_occ_mask(i)
    tracker.run(color, depth, reader.K, id_str, occ_mask=occ_mask)

  print("Done")
