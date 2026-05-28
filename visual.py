import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from models import BaseClassifier

# ==============================================================================
# [설정 변수 - ROOT 및 경로 설정]
# ==============================================================================
ROOT_DIR = "./baseline_CE"  # 모든 데이터가 들어있는 기본 폴더

# 경로 설정
METADATA_PATH  = "/home/ubuntu/MJ/dcase2026_task1_baseline/data/BSD10k-v1.2/metadata/BSD10k_metadata.csv"  
SPLIT_CSV_PATH = "/home/ubuntu/MJ/dcase2026_task1_baseline/model_output/both/fold_3/splits.csv"  # 파일명 splits.csv 확인
CKPT_PATH      = "/home/ubuntu/MJ/dcase2026_task1_baseline/model_output/both/fold_3/best_model.pth"
AUDIO_EMB_DIR  = "/home/ubuntu/MJ/dcase2026_task1_baseline/data/BSD10k-v1.2/features/clap_audio_embeddings"
TEXT_EMB_DIR   = "/home/ubuntu/MJ/dcase2026_task1_baseline/data/BSD10k-v1.2/features/clap_text_embeddings"
OUT_PNG_PATH   = os.path.join(ROOT_DIR, "embedding_comparison_analysis_test_only.png")

# t-SNE 및 시각화 설정
PERPLEXITY = 30.0
N_ITER = 1000
SEED = 42

if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR, exist_ok=True)
    print(f"📁 폴더가 없어서 새로 만들었습니다: {ROOT_DIR}")
# ==============================================================================

# 시각화 공통 함수
def run_tsne_and_plot(ax, vectors, df_meta, title, top_classes, sub_classes):
    tsne = TSNE(
    n_components=2, 
    perplexity=PERPLEXITY, 
    max_iter=N_ITER,   # n_iter를 max_iter로 변경
    random_state=SEED, 
    init="pca", 
    learning_rate="auto"
)
    vis_dims = tsne.fit_transform(np.vstack(vectors))

    temp_df = df_meta.copy()
    temp_df["x"], temp_df["y"] = vis_dims[:, 0], vis_dims[:, 1]

    colors = plt.cm.get_cmap("Set1", len(top_classes))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'P', '1', '2', '3', '4']
    sub_class_to_marker = {sub: markers[i % len(markers)] for i, sub in enumerate(sub_classes)}

    for i, tc in enumerate(top_classes):
        tc_df = temp_df[temp_df["top_class"] == tc]
        tc_color = colors(i)
        for sc in sorted(tc_df["sub_class"].unique()):
            sc_df = tc_df[tc_df["sub_class"] == sc]
            if len(sc_df) == 0: continue
            ax.scatter(sc_df["x"], sc_df["y"], color=tc_color, marker=sub_class_to_marker[sc],
                       label=f"{tc} | {sc}", alpha=0.6, s=60, edgecolors='white', linewidths=0.3)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.2)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 데이터 로드 및 테스트셋 필터링
    if not os.path.exists(METADATA_PATH) or not os.path.exists(SPLIT_CSV_PATH):
        return print("❌ 에러: 필수 데이터 파일(metadata 또는 splits.csv)이 없습니다.")

    df = pd.read_csv(METADATA_PATH)
    df.columns = df.columns.str.strip()

    # splits.csv 로드 및 테스트 ID 추출 (index 컬럼 사용)
    split_df = pd.read_csv(SPLIT_CSV_PATH)
    split_df.columns = split_df.columns.str.strip()

    # [수정] sound_id 대신 'index' 컬럼을 사용하여 테스트 샘플 ID 추출
    test_ids = set(split_df[split_df['split'] == 'test']['index'].astype(str).tolist())

    # 메타데이터에서 해당 index를 sound_id로 가진 테스트 샘플만 남기기
    df = df[df['sound_id'].astype(str).isin(test_ids)].reset_index(drop=True)
    print(f"✅ 테스트셋 샘플 수: {len(df)}개 필터링 완료.")

    # 모델 복원
    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = BaseClassifier(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    raw_embeddings, latent_vectors, meta_list = [], [], []

    # 2. 임베딩 추출
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Test Samples"):
            file_name = f"{str(row['sound_id'])}.npy"
            a_path, t_path = os.path.join(AUDIO_EMB_DIR, file_name), os.path.join(TEXT_EMB_DIR, file_name)
            a_tensor, t_tensor, raw_val = None, None, None

            if model.mode in ("audio", "both") and os.path.exists(a_path):
                raw_a = np.load(a_path).flatten()
                a_tensor = torch.tensor(raw_a, dtype=torch.float32, device=device).unsqueeze(0)
                raw_val = raw_a
            if model.mode in ("text", "both") and os.path.exists(t_path):
                raw_t = np.load(t_path).flatten()
                t_tensor = torch.tensor(raw_t, dtype=torch.float32, device=device).unsqueeze(0)
                if raw_val is None: raw_val = raw_t

            if (model.mode != "text" and a_tensor is None) or (model.mode != "audio" and t_tensor is None):
                continue

            z, _, _ = model(a_tensor, t_tensor)
            raw_embeddings.append(raw_val)
            latent_vectors.append(z.squeeze(0).cpu().numpy())
            meta_list.append({"top_class": row['class_top'], "sub_class": row['class']})

    if not latent_vectors: return print("❌ 추출된 데이터가 없습니다.")

    df_meta = pd.DataFrame(meta_list)
    top_classes = sorted(df_meta["top_class"].unique())
    sub_classes = sorted(df_meta["sub_class"].unique())

    # 3. 시각화
    fig, axes = plt.subplots(1, 2, figsize=(26, 12))
    run_tsne_and_plot(axes[0], raw_embeddings, df_meta, f"1. Raw CLAP (TEST ONLY, Mode: {model.mode})", top_classes, sub_classes)
    run_tsne_and_plot(axes[1], latent_vectors, df_meta, "2. Latent Features Z (TEST ONLY) after Training", top_classes, sub_classes)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=6, fontsize='x-small', title="Testset Hierarchy")

    plt.tight_layout()
    plt.savefig(OUT_PNG_PATH, dpi=200, bbox_inches='tight')
    print(f"✅ 저장 완료: {OUT_PNG_PATH}")
    plt.show()

if __name__ == "__main__":
    main()