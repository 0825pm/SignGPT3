"""
SOKE-style data loading functions for Sign Language
Supports both SOKE keys (smplx_*) and new keys (global_orient, body_pose, etc.)

Source: https://github.com/2000ZRL/SOKE/blob/main/mGPT/data/humanml/load_data.py
"""
import os
import math
import pickle
import numpy as np
from bisect import bisect_left, bisect_right


# ==============================================================================
# SMPL-X pose keys (두 가지 형식 모두 지원)
# ==============================================================================

# SOKE 원본 키 (179 dims)
SOKE_KEYS = [
    'smplx_root_pose',    # 3
    'smplx_body_pose',    # 63
    'smplx_lhand_pose',   # 45
    'smplx_rhand_pose',   # 45
    'smplx_jaw_pose',     # 3
    'smplx_shape',        # 10
    'smplx_expr'          # 10
]

# 새로운 키 형식 (일부 데이터셋)
NEW_KEYS = [
    'global_orient',      # 3  (= smplx_root_pose)
    'body_pose',          # 63 (= smplx_body_pose)
    'left_hand_pose',     # 45 (= smplx_lhand_pose)
    'right_hand_pose',    # 45 (= smplx_rhand_pose)
    'jaw_pose',           # 3  (= smplx_jaw_pose)
    'betas',              # 10 (= smplx_shape)
    'expression'          # 10 (= smplx_expr)
]

# 키 매핑 (new -> soke)
KEY_MAPPING = {
    'global_orient': 'smplx_root_pose',
    'body_pose': 'smplx_body_pose',
    'left_hand_pose': 'smplx_lhand_pose',
    'right_hand_pose': 'smplx_rhand_pose',
    'jaw_pose': 'smplx_jaw_pose',
    'betas': 'smplx_shape',
    'expression': 'smplx_expr'
}


def sample(input, count):
    """Uniform sampling"""
    ss = float(len(input)) / count
    return [input[int(math.floor(i * ss))] for i in range(count)]


def get_pose_from_pkl(poses_dict):
    """
    pkl 파일에서 pose 데이터 추출 (두 가지 키 형식 모두 지원)
    
    Returns:
        pose: numpy array of shape (179,) or None if failed
    """
    pose_values = []
    
    # 먼저 SOKE 키로 시도
    all_soke_keys_found = all(key in poses_dict for key in SOKE_KEYS)
    
    if all_soke_keys_found:
        # SOKE 키 사용
        for key in SOKE_KEYS:
            val = np.array(poses_dict[key]).flatten()
            pose_values.append(val)
    else:
        # 새 키로 시도 (순서 맞춰서)
        key_order = [
            ('global_orient', 'smplx_root_pose'),
            ('body_pose', 'smplx_body_pose'),
            ('left_hand_pose', 'smplx_lhand_pose'),
            ('right_hand_pose', 'smplx_rhand_pose'),
            ('jaw_pose', 'smplx_jaw_pose'),
            ('betas', 'smplx_shape'),
            ('expression', 'smplx_expr')
        ]
        
        for new_key, soke_key in key_order:
            # 새 키 또는 SOKE 키 중 있는 것 사용
            if new_key in poses_dict:
                val = np.array(poses_dict[new_key]).flatten()
                pose_values.append(val)
            elif soke_key in poses_dict:
                val = np.array(poses_dict[soke_key]).flatten()
                pose_values.append(val)
            else:
                # 키가 없으면 기본값
                print(f"Warning: Neither {new_key} nor {soke_key} found in pkl")
                return None
    
    if pose_values:
        pose = np.concatenate(pose_values)
        return pose
    
    return None


def convert_179_to_133(clip_poses):
    """
    179-dim → 133-dim 변환 (SOKE 방식)
    
    179-dim 구조:
    - root_pose: 0:3 (3)
    - body_pose: 3:66 (63 = 21 joints * 3)
    - lhand_pose: 66:111 (45 = 15 joints * 3)
    - rhand_pose: 111:156 (45 = 15 joints * 3)
    - jaw_pose: 156:159 (3)
    - shape: 159:169 (10)
    - expr: 169:179 (10)
    
    133-dim 구조 (after conversion):
    - upper_body: 0:30 (10 joints * 3, body joints 12-21)
    - lhand_pose: 30:75 (45)
    - rhand_pose: 75:120 (45)
    - jaw_pose: 120:123 (3)
    - expr: 123:133 (10)
    
    제거되는 부분:
    - root_pose (3) + lower_body (11 joints * 3 = 33) = 36 dims
    - shape (10 dims)
    """
    # Step 1: lower body 제거 (root 3 + 11 body joints * 3 = 36)
    clip_poses = clip_poses[:, (3 + 3 * 11):]  # 179 -> 143
    
    # Step 2: shape 제거하고 expr만 유지
    # 143 = 30(upper body) + 45(lhand) + 45(rhand) + 3(jaw) + 10(shape) + 10(expr)
    # [:-20] = 처음부터 shape 앞까지 (123)
    # [-10:] = expr만 (10)
    clip_poses = np.concatenate([clip_poses[:, :-20], clip_poses[:, -10:]], axis=1)  # 143 -> 133
    
    return clip_poses

def convert_179_to_120(clip_poses):
    """
    179-dim → 120-dim 변환 (upper_body + lhand + rhand only)
    
    179-dim에서 직접 추출:
    - [36:66]   upper_body (10 joints × 3 = 30)
    - [66:111]  lhand (15 joints × 3 = 45)
    - [111:156] rhand (15 joints × 3 = 45)
    
    Total: 30 + 45 + 45 = 120
    """
    return clip_poses[:, 36:156]

def load_h2s_sample(ann, data_dir, need_pose=True, code_path=None, need_code=False):
    """
    Load How2Sign sample
    
    Args:
        ann: dict with keys 'name', 'text', 'fps', 'split' (optional)
        data_dir: base directory for How2Sign data
        need_pose: whether to load pose data
        code_path: path to pre-computed codes
        need_code: whether to load codes
    
    Returns:
        clip_poses: numpy array of shape (T, 133)
        clip_text: text string
        name: sample name
        code: pre-computed code or None
    """
    name = ann['name']
    clip_text = ann['text']
    fps = ann.get('fps', 25)
    split = ann.get('split', 'train')
    
    # 경로 찾기 (여러 형식 지원)
    possible_paths = [
        os.path.join(data_dir, split, 'poses', name),
        os.path.join(data_dir, 'poses', name),
        os.path.join(data_dir, name),
    ]
    
    pose_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            pose_dir = path
            break
    
    if pose_dir is None:
        return None, None, None, None
    
    # 프레임 리스트 (여러 파일명 형식 지원)
    all_files = os.listdir(pose_dir)
    pkl_files = [f for f in all_files if f.endswith('.pkl')]
    
    if len(pkl_files) < 4:
        return None, None, None, None
    
    # 파일명 정렬 (숫자 순서)
    # 형식 1: name_0_3D.pkl, name_1_3D.pkl, ...
    # 형식 2: 0.pkl, 1.pkl, ...
    def get_frame_id(filename):
        try:
            # name_X_3D.pkl 형식
            if '_3D.pkl' in filename:
                parts = filename.replace('_3D.pkl', '').split('_')
                return int(parts[-1])
            # X.pkl 형식
            else:
                return int(filename.replace('.pkl', ''))
        except:
            return 0
    
    pkl_files = sorted(pkl_files, key=get_frame_id)
    
    # FPS 조정 (24fps로 리샘플링)
    if fps > 24:
        pkl_files = sample(pkl_files, count=int(24 * len(pkl_files) / fps))
    
    if len(pkl_files) < 4:
        return None, None, None, None
    
    # Pose 로딩
    clip_poses = np.zeros([len(pkl_files), 179], dtype=np.float32)
    
    if need_pose:
        for frame_id, frame_file in enumerate(pkl_files):
            frame_path = os.path.join(pose_dir, frame_file)
            try:
                with open(frame_path, 'rb') as f:
                    poses_dict = pickle.load(f)
                
                pose = get_pose_from_pkl(poses_dict)
                if pose is not None and len(pose) >= 179:
                    clip_poses[frame_id] = pose[:179]
                elif pose is not None:
                    clip_poses[frame_id, :len(pose)] = pose
                    
            except Exception as e:
                # 실패한 프레임은 0으로 유지
                continue
        
        # 179 -> 133 변환
        clip_poses = convert_179_to_120(clip_poses)
    else:
        clip_poses = np.zeros([len(pkl_files), 133], dtype=np.float32)
    
    # Code 로딩
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
    
    clip_poses = np.zeros([len(frame_list), 179], dtype=np.float32)
    
    if need_pose:
        for frame_id, frame in enumerate(frame_list):
            frame_path = os.path.join(pose_dir, frame)
            try:
                with open(frame_path, 'rb') as f:
                    poses_dict = pickle.load(f)
                
                pose = get_pose_from_pkl(poses_dict)
                if pose is not None and len(pose) >= 179:
                    clip_poses[frame_id] = pose[:179]
                elif pose is not None:
                    clip_poses[frame_id, :len(pose)] = pose
                    
            except:
                continue
        
        # 179 -> 133 변환
        clip_poses = convert_179_to_120(clip_poses)
    else:
        clip_poses = np.zeros([len(frame_list), 120], dtype=np.float32)
    
    # Code 로딩
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
    
    pose_dir = os.path.join(data_dir, name)
    if not os.path.exists(pose_dir):
        return None, None, None, None
    
    frame_list = sorted([f for f in os.listdir(pose_dir) if f.endswith('.pkl')])
    if len(frame_list) < 4:
        return None, None, None, None
    
    clip_poses = np.zeros([len(frame_list), 179], dtype=np.float32)
    
    if need_pose:
        for frame_id, frame in enumerate(frame_list):
            frame_path = os.path.join(pose_dir, frame)
            try:
                with open(frame_path, 'rb') as f:
                    poses_dict = pickle.load(f)
                
                pose = get_pose_from_pkl(poses_dict)
                if pose is not None and len(pose) >= 179:
                    clip_poses[frame_id] = pose[:179]
                elif pose is not None:
                    clip_poses[frame_id, :len(pose)] = pose
                    
            except:
                continue
        
        # 179 -> 133 변환
        clip_poses = convert_179_to_120(clip_poses)
    else:
        clip_poses = np.zeros([len(frame_list), 120], dtype=np.float32)
    
    # Code 로딩
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
    
    # Frame index 추출 및 범위 선택
    def get_frame_idx(filename):
        try:
            if dataset == 'phoenix_iso':
                return int(filename.replace('.pkl', '').replace('images', ''))
            else:
                return int(filename.split('.pkl')[0])
        except:
            return 0
    
    frame_idx = [get_frame_idx(f) for f in frame_list]
    
    if start >= 0 and end > 0:
        start_idx = bisect_left(frame_idx, start)
        end_idx = bisect_right(frame_idx, end)
        frame_list = frame_list[start_idx:end_idx]
        
        if len(frame_list) < 4:
            return None, None, None, None
        
        ratio = len(frame_list) / (end - start)
        if ratio < 0.5:
            return None, None, None, None
    
    clip_poses = np.zeros([len(frame_list), 179], dtype=np.float32)
    
    if need_pose:
        for frame_id, frame in enumerate(frame_list):
            frame_path = os.path.join(pose_dir, frame)
            try:
                with open(frame_path, 'rb') as f:
                    poses_dict = pickle.load(f)
                
                pose = get_pose_from_pkl(poses_dict)
                if pose is not None and len(pose) >= 179:
                    clip_poses[frame_id] = pose[:179]
                elif pose is not None:
                    clip_poses[frame_id, :len(pose)] = pose
                    
            except:
                continue
        
        # 179 -> 133 변환
        clip_poses = convert_179_to_120(clip_poses)
    else:
        clip_poses = np.zeros([len(frame_list), 120], dtype=np.float32)
    
    # Code 로딩
    code = None
    if need_code and code_path:
        try:
            if dataset == 'csl_iso':
                fname = os.path.join(code_path, 'csl', f'{name}.npy')
            elif dataset == 'phoenix_iso':
                fname = os.path.join(code_path, 'phoenix', f'{name}.npy')
            elif dataset == 'how2sign_iso':
                fname = os.path.join(code_path, 'how2sign', f'{name}.npy')
            else:
                fname = os.path.join(code_path, f'{name}.npy')
            
            if not os.path.exists(fname):
                fname = os.path.join(code_path, f'{name}.npy')
            code = np.load(fname)[0]
        except:
            pass
    
    return clip_poses.astype(np.float32), clip_text, name, code


# Backward compatibility
keys = SOKE_KEYS  # 기존 코드와의 호환성 유지