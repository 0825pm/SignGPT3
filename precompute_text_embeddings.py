"""
Text Embedding 전처리 스크립트

mBART Encoder로 모든 텍스트를 미리 인코딩하여 저장.
학습 시 mBART forward 생략 → 속도/메모리 향상

★★★ 중요: 기존 SignGPT3 Dataset 클래스의 annotation 로딩 로직을 재사용 ★★★

사용법:
    python precompute_text_embeddings.py \
        --mbart_path ./deps/mbart-h2s-csl-phoenix \
        --data_root /path/to/How2Sign \
        --csl_root /path/to/CSL-Daily \
        --phoenix_root /path/to/Phoenix_2014T \
        --output_dir ./cached_embeddings \
        --batch_size 64

출력:
    ./cached_embeddings/
        ├── how2sign/           ← src별로 분리 저장!
        │   ├── train/
        │   │   └── {name}.pt
        │   ├── val/
        │   └── test/
        ├── csl/
        └── phoenix/

복사 위치: SignGPT3/precompute_text_embeddings.py
"""

import os
import sys
import argparse
import gzip
import pickle
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import MBartModel, MBartTokenizer


# Bad How2Sign IDs (from SOKE)
BAD_H2S_IDS = [
    '0DU7wWLK-QU_0-8-rgb_front', '0ICZi26jdaQ_28-5-rgb_front', '0vNfEYst_tQ_11-8-rgb_front',
    '13X0vEMNm7M_8-5-rgb_front', '14weIYQswlE_23-8-rgb_front', '1B56XMJ-j1Q_13-8-rgb_front',
    '1P0oKY4FNyI_0-8-rgb_front', '1dpRaxOTfZs_0-8-rgb_front', '1ei1kVTw23A_29-8-rgb_front',
    '1spCnuBmWYk_0-8-rgb_front', '2-vXO7MMLJc_0-5-rgb_front', '21PbS6wnHtY_0-5-rgb_front',
    '3tyfxL2wO-M_0-8-rgb_front', 'BpYDl3AO4B8_0-1-rgb_front', 'CH7AviIr0-0_14-8-rgb_front',
    'CJ8RyW9pzKU_6-8-rgb_front', 'D0T7ho08Q3o_25-2-rgb_front', 'Db5SUQvNsHc_18-1-rgb_front',
    'Eh697LCFjTw_0-3-rgb_front', 'F-p1IdedNbg_23-8-rgb_front', 'aUBQCNegrYc_13-1-rgb_front',
    'cvn7htBA8Xc_9-8-rgb_front', 'czBrBQgZIuc_19-5-rgb_front', 'dbSAB8F8GYc_11-9-rgb_front',
    'doMosV-zfCI_7-2-rgb_front', 'dvBdWGLzayI_10-8-rgb_front', 'eBrlZcccILg_26-3-rgb_front',
    '39FN42e41r0_17-1-rgb_front', 'a4Nxq0QV_WA_9-3-rgb_front', 'fzrJBu2qsM8_11-8-rgb_front',
    'g3Cc_1-V31U_12-3-rgb_front',
]


# =============================================================================
# Data Loading (SOKE style - 기존 코드와 동일!)
# =============================================================================

def load_annotations(data_root, csl_root, phoenix_root, split, dataset_name='how2sign_csl_phoenix'):
    """
    Load annotations from all datasets
    ★ 기존 SignGPT3/SOKE 코드의 annotation 로딩 로직과 동일하게 구현
    """
    all_data = []
    stats = {'how2sign': 0, 'csl': 0, 'phoenix': 0}
    
    # =========================================================================
    # How2Sign (영어)
    # 경로: {root}/{split}/re_aligned/how2sign_realigned_{split}_preprocessed_fps.csv
    # =========================================================================
    if 'how2sign' in dataset_name:
        csv_path = os.path.join(data_root, split, 're_aligned', 
                                f'how2sign_realigned_{split}_preprocessed_fps.csv')
        if os.path.exists(csv_path):
            csv_data = pd.read_csv(csv_path)
            for _, row in csv_data.iterrows():
                name = row['SENTENCE_NAME']
                if name in BAD_H2S_IDS:
                    continue
                all_data.append({
                    'name': name,
                    'text': row['SENTENCE'],
                    'src': 'how2sign',
                    'split': split,
                })
            stats['how2sign'] = len([d for d in all_data if d['src'] == 'how2sign'])
            print(f"  [{split}] How2Sign: {stats['how2sign']} samples (from {csv_path})")
        else:
            print(f"  [{split}] How2Sign: CSV not found at {csv_path}")
    
    # =========================================================================
    # CSL-Daily (중국어)
    # 경로: {csl_root}/csl_clean.{split}  ★ 수정됨!
    # =========================================================================
    if 'csl' in dataset_name and csl_root:
        # 여러 가능한 경로 시도 (우선순위 순)
        possible_paths = [
            os.path.join(csl_root, f'csl_clean.{split}'),  # ★ SOKE 기본 형식
            os.path.join(csl_root, 'annotations', f'csl_clean.{split}'),
            os.path.join(csl_root, 'annotations', f'csl_{split}.pkl.gz'),
            os.path.join(csl_root, f'csl_{split}.pkl.gz'),
        ]
        
        ann_path = None
        for p in possible_paths:
            if os.path.exists(p):
                ann_path = p
                break
        
        if ann_path:
            with gzip.open(ann_path, 'rb') as f:
                ann = pickle.load(f)
            for item in ann:
                item_copy = deepcopy(item)
                item_copy['src'] = 'csl'
                item_copy['split'] = split
                all_data.append(item_copy)
            stats['csl'] = len(ann)
            print(f"  [{split}] CSL-Daily: {stats['csl']} samples (from {ann_path})")
        else:
            print(f"  [{split}] CSL-Daily: Annotation not found. Tried: {possible_paths}")
    
    # =========================================================================
    # Phoenix-2014T (독일어)
    # 경로: {phoenix_root}/phoenix14t.{split}  ★ val → dev 변환!
    # =========================================================================
    if 'phoenix' in dataset_name and phoenix_root:
        # val → dev 변환 (Phoenix 특수 케이스)
        phoenix_split = 'dev' if split == 'val' else split
        
        possible_paths = [
            os.path.join(phoenix_root, f'phoenix14t.{phoenix_split}'),  # ★ SOKE 기본 형식
            os.path.join(phoenix_root, 'annotations', f'phoenix14t.{phoenix_split}'),
            os.path.join(phoenix_root, 'annotations', f'phoenix_{split}.pkl.gz'),
            os.path.join(phoenix_root, f'phoenix_{split}.pkl.gz'),
        ]
        
        ann_path = None
        for p in possible_paths:
            if os.path.exists(p):
                ann_path = p
                break
        
        if ann_path:
            with gzip.open(ann_path, 'rb') as f:
                ann = pickle.load(f)
            for item in ann:
                item_copy = deepcopy(item)
                item_copy['src'] = 'phoenix'
                item_copy['split'] = split
                all_data.append(item_copy)
            stats['phoenix'] = len(ann)
            print(f"  [{split}] Phoenix: {stats['phoenix']} samples (from {ann_path})")
        else:
            print(f"  [{split}] Phoenix: Annotation not found. Tried: {possible_paths}")
    
    print(f"  [{split}] Total: {len(all_data)} (H2S:{stats['how2sign']}, CSL:{stats['csl']}, PHX:{stats['phoenix']})")
    return all_data


def get_src_lang(src):
    """Get mBART language token for source dataset"""
    return {'how2sign': 'en_XX', 'csl': 'zh_CN', 'phoenix': 'de_DE'}.get(src, 'en_XX')


def sanitize_filename(name):
    """파일명에 사용할 수 없는 문자 치환"""
    # '/', '\', ':', '*', '?', '"', '<', '>', '|' 등 제거
    return name.replace('/', '_').replace('\\', '_').replace(':', '_')


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Precompute mBART text embeddings')
    parser.add_argument('--mbart_path', type=str, default='./deps/mbart-h2s-csl-phoenix')
    parser.add_argument('--data_root', type=str, required=True, help='How2Sign root')
    parser.add_argument('--csl_root', type=str, default=None)
    parser.add_argument('--phoenix_root', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./cached_embeddings')
    parser.add_argument('--dataset_name', type=str, default='how2sign_csl_phoenix')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--proj_dim', type=int, default=768, help='Project to this dim (0=no proj)')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # ==========================================================================
    # 1. Load mBART
    # ==========================================================================
    print(f"\n[1/3] Loading mBART from {args.mbart_path}")
    tokenizer = MBartTokenizer.from_pretrained(args.mbart_path)
    tokenizer.add_tokens(['en_ASL', 'zh_CSL', 'de_DGS'], special_tokens=True)
    
    mbart = MBartModel.from_pretrained(args.mbart_path)
    mbart.resize_token_embeddings(len(tokenizer))
    encoder = mbart.encoder.to(device).eval()
    mbart_dim = mbart.config.d_model  # 1024
    
    # Optional projection layer
    proj = None
    if args.proj_dim > 0 and args.proj_dim != mbart_dim:
        proj = nn.Sequential(
            nn.Linear(mbart_dim, args.proj_dim),
            nn.LayerNorm(args.proj_dim),
        ).to(device)
        print(f"  Projection: {mbart_dim} → {args.proj_dim}")
    
    print(f"  mBART hidden dim: {mbart_dim}")
    print(f"  Tokenizer vocab size: {len(tokenizer)}")
    
    # ==========================================================================
    # 2. Process each split
    # ==========================================================================
    total_saved = 0
    
    for split in args.splits:
        print(f"\n[2/3] Processing {split} split...")
        
        # Load annotations
        all_data = load_annotations(
            args.data_root, args.csl_root, args.phoenix_root,
            split, args.dataset_name
        )
        
        if len(all_data) == 0:
            print(f"  No data found for {split}, skipping.")
            continue
        
        # ★★★ src별로 그룹화 ★★★
        data_by_src = {}
        for item in all_data:
            src = item['src']
            if src not in data_by_src:
                data_by_src[src] = []
            data_by_src[src].append(item)
        
        # Process each source dataset separately
        for src, src_data in data_by_src.items():
            print(f"\n  Processing {src} ({len(src_data)} samples)...")
            
            # ★★★ 저장 경로: {output_dir}/{src}/{split}/ ★★★
            output_src_split_dir = Path(args.output_dir) / src / split
            output_src_split_dir.mkdir(parents=True, exist_ok=True)
            
            # Process in batches
            num_batches = (len(src_data) + args.batch_size - 1) // args.batch_size
            
            for batch_idx in tqdm(range(num_batches), desc=f"    {src}/{split}"):
                start_idx = batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(src_data))
                batch_data = src_data[start_idx:end_idx]
                
                texts = [d['text'] for d in batch_data]
                names = [d['name'] for d in batch_data]
                
                # Tokenize
                encoding = tokenizer(
                    texts,
                    padding='longest',
                    max_length=args.max_length,
                    truncation=True,
                    return_tensors='pt',
                )
                input_ids = encoding.input_ids.to(device)
                attention_mask = encoding.attention_mask.to(device)
                
                # Set correct language tokens (all same src in this batch)
                lang_id = tokenizer.convert_tokens_to_ids(get_src_lang(src))
                for i in range(len(batch_data)):
                    seq_len = int(attention_mask[i].sum().item())
                    input_ids[i, seq_len - 1] = lang_id
                
                # Encode
                with torch.no_grad():
                    outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
                    embeddings = outputs.last_hidden_state  # [B, seq, 1024]
                    
                    # Project if needed
                    if proj is not None:
                        embeddings = proj(embeddings)  # [B, seq, 768]
                
                # Save each sample
                for i, name in enumerate(names):
                    seq_len = int(attention_mask[i].sum().item())
                    emb = embeddings[i, :seq_len].cpu()  # [seq_len, dim]
                    mask = attention_mask[i, :seq_len].cpu()  # [seq_len]
                    
                    # Sanitize filename
                    safe_name = sanitize_filename(name)
                    save_path = output_src_split_dir / f"{safe_name}.pt"
                    
                    torch.save({
                        'emb': emb,
                        'mask': mask,
                        'src': src,
                        'text': texts[i],  # Keep original text for debugging
                        'name': name,      # Original name
                    }, save_path)
            
            total_saved += len(src_data)
            print(f"    Saved {len(src_data)} embeddings to {output_src_split_dir}")
    
    # ==========================================================================
    # 3. Summary
    # ==========================================================================
    print(f"\n[3/3] Done!")
    print(f"  Total samples: {total_saved}")
    print(f"  Output directory: {args.output_dir}")
    
    # Directory structure
    print(f"\n  📁 Directory structure:")
    print(f"  {args.output_dir}/")
    for src in ['how2sign', 'csl', 'phoenix']:
        src_dir = Path(args.output_dir) / src
        if src_dir.exists():
            for split_dir in sorted(src_dir.iterdir()):
                if split_dir.is_dir():
                    count = len(list(split_dir.glob('*.pt')))
                    print(f"    ├── {src}/{split_dir.name}/ ({count} files)")
    
    # Check total size
    total_size = sum(f.stat().st_size for f in Path(args.output_dir).rglob('*.pt'))
    print(f"\n  Total size: {total_size / 1e9:.2f} GB")


if __name__ == '__main__':
    main()