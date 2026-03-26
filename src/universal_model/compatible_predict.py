import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def find_image_path(image_dir, sid):
    """
    Glob-based flexible image matching to handle different extensions and suffixes.
    """
    if not image_dir or not os.path.exists(image_dir):
        return None
    # 1. Exact match .jpg
    p1 = os.path.join(image_dir, f"{sid}.jpg")
    if os.path.exists(p1): return p1
    # 2. Exact match .bmp
    p2 = os.path.join(image_dir, f"{sid}.bmp")
    if os.path.exists(p2): return p2
    # 3. Pattern match like sid_*.bmp or sid_*.jpg
    matches = glob.glob(os.path.join(image_dir, f"{sid}*.*"))
    if matches:
        return matches[0]
    return None

def extract_image_features(image_path):
    """
    Extract robust physical features from RGB images to supplement spectral features.
    """
    try:
        if not image_path or not os.path.exists(image_path):
            return np.zeros(6)

        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img_small = img.resize((64, 64))
            arr = np.array(img_small)
            
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            
            features = [
                np.mean(r), np.std(r),
                np.mean(g), np.std(g),
                np.mean(b), np.std(b)
            ]
            return np.array(features)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return np.zeros(6)

def resample_spectra(spectral_data, original_dim, target_dim=500):
    x_original = np.linspace(0, 1, original_dim)
    x_target = np.linspace(0, 1, target_dim)
    
    resampled_data = []
    for row in spectral_data:
        f = interp1d(x_original, row, kind='linear', fill_value="extrapolate")
        resampled_data.append(f(x_target))
        
    return np.array(resampled_data)

def load_dataset_universal(config):
    print(f"Loading dataset: {config['excel']}")
    df = pd.read_excel(config['excel'], header=config['header'])
    
    # Extract IDs
    if config['id_col'] is None:
        sample_ids = np.array([f"{config.get('prefix', '')}{i+1}" for i in range(len(df))])
    else:
        sample_ids = df.iloc[:, config['id_col']].astype(str).values
        
    # Extract Targets
    target_idx = config['target_col']
    df.iloc[:, target_idx] = pd.to_numeric(df.iloc[:, target_idx], errors='coerce')
    
    # Drop NaNs based on target
    valid_mask = df.iloc[:, target_idx].notna()
    df = df[valid_mask]
    sample_ids = sample_ids[valid_mask.values]
    targets = df.iloc[:, target_idx].astype(float).values
    
    # Extract Spectra
    spec_start = config['spec_start_col']
    spectra = df.iloc[:, spec_start:].apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    original_dim = spectra.shape[1]
    resampled_spectra = resample_spectra(spectra, original_dim, target_dim=500)
    
    print(f"  -> Found {len(sample_ids)} valid samples.")
    print(f"  -> Resampled spectrum from {original_dim} to 500 dimensions.")

    return sample_ids, targets, resampled_spectra, config['images']


def evaluate_universal_system(sample_ids, targets, spectra, image_dir, dataset_name, save_dir):
    """
    Execute the late-fusion pipeline and generate validation plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Extract Image Features
    print(f"Extracting image features for {dataset_name}...")
    image_features_list = []
    found_images = 0
    for sid in sample_ids:
        img_path = find_image_path(image_dir, sid)
        if img_path:
            found_images += 1
        feats = extract_image_features(img_path)
        image_features_list.append(feats)
    print(f"  -> Found {found_images}/{len(sample_ids)} images for feature extraction.")
    image_feats = np.array(image_features_list)
    
    # 2. Cross Validation Setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    y_true_all = []
    y_pred_spec_all = []
    y_pred_img_all = []
    
    print("Running 5-Fold Cross Validation...")
    
    for train_idx, val_idx in kf.split(sample_ids):
        # -----------------------------------
        # Branch 1: Spectrum PLSR
        # -----------------------------------
        X_spec_train, X_spec_val = spectra[train_idx], spectra[val_idx]
        y_train, y_val = targets[train_idx], targets[val_idx]
        
        pls = PLSRegression(n_components=min(10, len(train_idx)))
        pls.fit(X_spec_train, y_train)
        pred_spec = pls.predict(X_spec_val).ravel()
        
        # -----------------------------------
        # Branch 2: Image Random Forest
        # -----------------------------------
        X_img_train, X_img_val = image_feats[train_idx], image_feats[val_idx]
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_img_train, y_train)
        pred_img = rf.predict(X_img_val)
        
        y_true_all.extend(y_val)
        y_pred_spec_all.extend(pred_spec)
        y_pred_img_all.extend(pred_img)

    y_true_all = np.array(y_true_all)
    y_pred_spec_all = np.array(y_pred_spec_all)
    y_pred_img_all = np.array(y_pred_img_all)
    
    # 3. Dynamic Late Fusion Optimization (Find best Alpha)
    best_alpha = 0.0
    best_r2 = -float('inf')
    predictions_history = {}
    
    for alpha in np.linspace(0, 1, 101):
        fused_pred = alpha * y_pred_spec_all + (1 - alpha) * y_pred_img_all
        current_r2 = r2_score(y_true_all, fused_pred)
        if current_r2 > best_r2:
            best_r2 = current_r2
            best_alpha = alpha
            
    best_fused_pred = best_alpha * y_pred_spec_all + (1 - best_alpha) * y_pred_img_all
    
    # 4. Generate Visualization (Loss/Metrics equivalent mapping)
    r2_spec = r2_score(y_true_all, y_pred_spec_all)
    r2_img = r2_score(y_true_all, y_pred_img_all)
    r2_fusion = best_r2
    
    rmse_spec = np.sqrt(mean_squared_error(y_true_all, y_pred_spec_all))
    rmse_img = np.sqrt(mean_squared_error(y_true_all, y_pred_img_all))
    rmse_fusion = np.sqrt(mean_squared_error(y_true_all, best_fused_pred))

    print(f"\n--- {dataset_name} Results ---")
    print(f"Optimal Spec Weight (Alpha): {best_alpha:.2f}")
    print(f"Spec Branch R2: {r2_spec:.4f} | RMSE: {rmse_spec:.4f}")
    print(f"Img  Branch R2: {r2_img:.4f}  | RMSE: {rmse_img:.4f}")
    print(f"Fused Model R2: {r2_fusion:.4f} | RMSE: {rmse_fusion:.4f}")

    # Plot Scatter plots (Replacing typical "Loss" curve for ML paradigms)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = [f'光谱分支预测 (R2={r2_spec:.2f})', f'图像分支预测 (R2={r2_img:.2f})', f'跨模态自适应融合 (R2={r2_fusion:.2f})']
    preds = [y_pred_spec_all, y_pred_img_all, best_fused_pred]
    colors = ['skyblue', 'lightgreen', 'salmon']
    
    for ax, pred, title, c in zip(axes, preds, titles, colors):
        ax.scatter(y_true_all, pred, c=c, alpha=0.7, s=50)
        vmin = min(np.min(y_true_all), np.min(pred))
        vmax = max(np.max(y_true_all), np.max(pred))
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=2)
        ax.set_xlabel('真实 SOM')
        ax.set_ylabel('预测 SOM')
        ax.set_title(title)
        ax.grid(alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_prediction_scatter.png'), dpi=300)
    plt.close()
    
    return {
        'dataset': dataset_name,
        'alpha': best_alpha,
        'r2_fusion': r2_fusion,
        'rmse_fusion': rmse_fusion
    }


if __name__ == "__main__":
    work_dir = "work_save/universal_benchmark_fixed_v"
    os.makedirs(work_dir, exist_ok=True)
    
    print("===================================================")
    print(" 统一架构相兼性验证测试 (Universal Framework Validation)")
    print(" 核心机制：插值波段对齐 + 图像宏观光度特征 + 自适应晚期融合")
    print("===================================================\n")
    
    config_train = {
        "excel": "train_data/ref1.xlsx",
        "images": "train_data/images1",
        "header": None,
        "id_col": 0,
        "target_col": 1,
        "spec_start_col": 2,
    }
    
    config_new = {
        "excel": "new_data/ref2.xlsx",
        "images": "new_data/images2",
        "header": 0,
        "id_col": None,
        "target_col": 0,
        "spec_start_col": 1,
        "prefix": "c"
    }

    config_final = {
        "excel": "final_data/final.xlsx",
        "images": "final_data/images3",
        "header": 0,
        "id_col": 0,
        "target_col": 1,
        "spec_start_col": 2,
    }
    
    # 1. Train 数据集
    t_ids, t_targets, t_spec, t_img_dir = load_dataset_universal(config_train)
    evaluate_universal_system(t_ids, t_targets, t_spec, t_img_dir, "Train_Dataset", work_dir)
    
    # 2. New 数据集
    n_ids, n_targets, n_spec, n_img_dir = load_dataset_universal(config_new)
    evaluate_universal_system(n_ids, n_targets, n_spec, n_img_dir, "New_Dataset", work_dir)
    
    # 3. Final 数据集
    f_ids, f_targets, f_spec, f_img_dir = load_dataset_universal(config_final)
    evaluate_universal_system(f_ids, f_targets, f_spec, f_img_dir, "Final_Dataset", work_dir)
    
    print(f"\n所有图表和输出已保存至: {work_dir}")
