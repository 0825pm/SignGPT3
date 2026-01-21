"""
SOKE-style data loading functions
Source: https://github.com/2000ZRL/SOKE/blob/main/mGPT/data/humanml/load_data.py
"""
import os
import math
import pickle
import numpy as np
from bisect import bisect_left, bisect_right


# SMPL-X pose keys
keys = ['global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 
        'left_hand_pose', 'right_hand_pose', 'transl', 'betas', 'expression']


def sample(input, count):
    """Uniform sampling"""
    ss = float(len(input)) / count
    return [input[int(math.floor(i * ss))] for i in range(count)]


def load_h2s_sample(ann, data_dir, need_pose=True, code_path=None, need_code=False):
    """Load How2Sign sample"""
    clip_text = ann['text']
    name = ann['name']
    split = ann.get('split', 'train')
    
    pose_dir = os.path.join(data_dir, split, 'poses', name)
    if not os.path.exists(pose_dir):
        pose_dir = os.path.join(data_dir, 'poses', name)
    if not os.path.exists(pose_dir):
        return None, None, None, None
    
    frame_list = sorted([f for f in os.listdir(pose_dir) if f.endswith('.pkl')])
    if len(frame_list) < 4:
        return None, None, None, None
    
    clip_poses = np.zeros([len(frame_list), 179])
    if need_pose:
        for frame_id, frame in enumerate(frame_list):
            frame_path = os.path.join(pose_dir, frame)
            try:
                with open(frame_path, 'rb') as f:
                    poses = pickle.load(f)
                pose = np.concatenate([poses[key] for key in keys if key in poses], 0)
                clip_poses[frame_id] = pose[:179] if len(pose) >= 179 else np.pad(pose, (0, 179-len(pose)))
            except:
                continue
        
        # Convert 179-dim to 133-dim (SOKE format)
        clip_poses = clip_poses[:, (3 + 3 * 11):]  # Remove global + first 11 body joints
        clip_poses = np.concatenate([clip_poses[:, :-20], clip_poses[:, -10:]], axis=1)  # 179-36-10=133
    
    code = None
    if need_code and code_path:
        try:
            fname = os.path.join(code_path, 'how2sign', f'{name}.npy')
            if not os.path.exists(fname):
                fname = os.path.join(code_path, f'{name}.npy')
            code = np.load(fname)[0]
        except:
            pass
    
    return clip_poses.astype(np.float32), clip_text, name, code


def load_csl_sample(ann, data_dir, need_pose=True, code_path=None, need_code=False):
    """Load CSL-Daily sample"""
    clip_text = ann['text']
    name = ann['name']
    
    pose_dir = os.path.join(data_dir, 'poses', name)
    if not os.path.exists(pose_dir):
        return None, None, None, None
    
    frame_list = sorted([f for f in os.listdir(pose_dir) if f.endswith('.pkl')])
    if len(frame_list) < 4:
        return None, None, None, None
    
    clip_poses = np.zeros([len(frame_list), 179])
    if need_pose:
        for frame_id, frame in enumerate(frame_list):
            frame_path = os.path.join(pose_dir, frame)
            try:
                with open(frame_path, 'rb') as f:
                    poses = pickle.load(f)
                pose = np.concatenate([poses[key] for key in keys if key in poses], 0)
                clip_poses[frame_id] = pose[:179] if len(pose) >= 179 else np.pad(pose, (0, 179-len(pose)))
            except:
                continue
        
        clip_poses = clip_poses[:, (3 + 3 * 11):]
        clip_poses = np.concatenate([clip_poses[:, :-20], clip_poses[:, -10:]], axis=1)
    
    code = None
    if need_code and code_path:
        try:
            fname = os.path.join(code_path, 'csl', f'{name}.npy')
            if not os.path.exists(fname):
                fname = os.path.join(code_path, f'{name}.npy')
            code = np.load(fname)[0]
        except:
            pass
    
    return clip_poses.astype(np.float32), clip_text, name, code


def load_phoenix_sample(ann, data_dir, need_pose=True, code_path=None, need_code=False):
    """Load Phoenix-2014T sample"""
    clip_text = ann['text']
    name = ann['name']
    
    # Phoenix stores frames directly in split folder
    frame_list = sorted([f for f in os.listdir(os.path.join(data_dir, name)) if f.endswith('.pkl')])
    if len(frame_list) < 4:
        return None, None, None, None
    
    clip_poses = np.zeros([len(frame_list), 179])
    if need_pose:
        for frame_id, frame in enumerate(frame_list):
            frame_path = os.path.join(data_dir, name, frame)
            try:
                with open(frame_path, 'rb') as f:
                    poses = pickle.load(f)
                pose = np.concatenate([poses[key] for key in keys if key in poses], 0)
                clip_poses[frame_id] = pose[:179] if len(pose) >= 179 else np.pad(pose, (0, 179-len(pose)))
            except:
                continue
        
        clip_poses = clip_poses[:, (3 + 3 * 11):]
        clip_poses = np.concatenate([clip_poses[:, :-20], clip_poses[:, -10:]], axis=1)
    
    code = None
    if need_code and code_path:
        try:
            fname = os.path.join(code_path, 'phoenix', f'{name}.npy')
            if not os.path.exists(fname):
                fname = os.path.join(code_path, f'{name}.npy')
            code = np.load(fname)[0]
        except:
            pass
    
    return clip_poses.astype(np.float32), clip_text, name, code


def load_iso_sample(ann, data_dir, need_pose=True, code_path=None, need_code=False, dataset=None):
    """Load isolated sign sample"""
    clip_text = ann.get('label', '')
    name = ann['name']
    start, end = ann.get('start', 0), ann.get('end', -1)
    video_file = ann.get('video_file', name)
    
    if dataset in ['csl_iso', 'how2sign_iso']:
        pose_dir = os.path.join(data_dir, 'poses', video_file)
    elif dataset == 'phoenix_iso':
        pose_dir = os.path.join(data_dir, video_file)
    else:
        return None, None, None, None
    
    if not os.path.exists(pose_dir):
        return None, None, None, None
    
    frame_list = sorted([f for f in os.listdir(pose_dir) if f.endswith('.pkl')])
    if len(frame_list) < 4:
        return None, None, None, None
    
    # Get frame indices and select range
    if dataset in ['csl_iso', 'how2sign_iso']:
        frame_idx = [int(x.split('.pkl')[0]) for x in frame_list]
    elif dataset == 'phoenix_iso':
        frame_idx = [int(x.split('.pkl')[0].replace('images', '')) for x in frame_list]
    
    if end > 0:
        start_idx = bisect_left(frame_idx, start)
        end_idx = bisect_right(frame_idx, end)
        frame_list = frame_list[start_idx:end_idx]
        
        ratio = len(frame_list) / (end - start) if (end - start) > 0 else 0
        if ratio < 0.5 or len(frame_list) < 4:
            return None, None, None, None
    
    clip_poses = np.zeros([len(frame_list), 179])
    if need_pose:
        for frame_id, frame in enumerate(frame_list):
            if dataset in ['csl_iso', 'how2sign_iso']:
                frame_path = os.path.join(data_dir, 'poses', video_file, frame)
            else:
                frame_path = os.path.join(data_dir, video_file, frame)
            
            try:
                with open(frame_path, 'rb') as f:
                    poses = pickle.load(f)
                pose = np.concatenate([poses[key] for key in keys if key in poses], 0)
                clip_poses[frame_id] = pose[:179] if len(pose) >= 179 else np.pad(pose, (0, 179-len(pose)))
            except:
                continue
        
        clip_poses = clip_poses[:, (3 + 3 * 11):]
        clip_poses = np.concatenate([clip_poses[:, :-20], clip_poses[:, -10:]], axis=1)
    
    code = None
    if need_code and code_path:
        try:
            if dataset == 'csl_iso':
                fname = os.path.join(code_path, 'csl', f'{name}.npy')
            elif dataset == 'phoenix_iso':
                fname = os.path.join(code_path, 'phoenix', f'{name}.npy')
            else:
                fname = os.path.join(code_path, 'how2sign', f'{name}.npy')
            if not os.path.exists(fname):
                fname = os.path.join(code_path, f'{name}.npy')
            code = np.load(fname)[0]
        except:
            pass
    
    return clip_poses.astype(np.float32), clip_text, name, code