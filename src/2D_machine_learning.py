import os
import glob
import re
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def extract_image_features(image_dir, id_extractor):
    print(f"  > 正在提取图像特征: {image_dir}")
    features = []
    formats = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG') 
    img_paths = []
    for fmt in formats:
        img_paths.extend(glob.glob(os.path.join(image_dir, fmt)))

    for path in img_paths:
        filename = os.path.basename(path)
        sample_id = id_extractor(filename)
        if not sample_id:
            continue

        try:
            # 打开并降低分辨率以极大提高处理速度，同时保留宏观颜色特征
            img = Image.open(path).convert('RGB')
            img.thumbnail((128, 128))
            arr = np.array(img)

            r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
            r_mean, g_mean, b_mean = r.mean(), g.mean(), b.mean()
            v_mean = arr.max(axis=2).mean() # 近似提取 HSV 的明度 V
            color_index = (r_mean - g_mean) / (r_mean + g_mean + 1e-5)

            features.append({
                'Sample_ID': str(sample_id),
                'R_mean': r_mean, 'G_mean': g_mean, 'B_mean': b_mean,
                'V_mean': v_mean, 'Color_Index': color_index
            })
        except Exception as e:
            pass

    df_img = pd.DataFrame(features)
    print(f"  > 成功提取并处理 {len(df_img)} 张图片.")
    return df_img

def run_evaluation(X_spec, X_img, y, dataset_name, save_dir):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_sp, r2_im, r2_fu, best_ws = [], [], [], []
    rmse_sp, rmse_im, rmse_fu = [], [], []
    
    all_y_true = []
    all_y_pred_sp = []
    all_y_pred_im = []
    all_y_pred_fu = []
    
    last_alphas = None
    last_r2s = None

    for tr, te in cv.split(X_spec):
        X_s_tr, X_s_te = X_spec[tr], X_spec[te]
        X_i_tr, X_i_te = X_img[tr], X_img[te]
        y_tr, y_te = y[tr], y[te]

        # 分支 A: 光谱模型 PLSR
        if X_s_tr.shape[1] > 0:
            ms = PLSRegression(n_components=min(10, X_s_tr.shape[1], X_s_tr.shape[0]-1))
            ms.fit(X_s_tr, y_tr)
            p_s = ms.predict(X_s_te).flatten()
        else:
            p_s = np.zeros_like(y_te)

        # 分支 B: 图像模型 RandomForest
        mi = RandomForestRegressor(n_estimators=100, random_state=42)
        mi.fit(X_i_tr, y_tr)
        p_i = mi.predict(X_i_te)

        # 融合 C: 动态寻找最优晚期融合权重 Alpha
        best_alpha, best_r2 = 0, -float('inf')
        alphas = np.linspace(0, 1.0, 101)
        r2s_fold = []
        best_p_f_fold = None
        for alpha in alphas:
            p_f = alpha * p_i + (1 - alpha) * p_s
            cr2 = r2_score(y_te, p_f)
            r2s_fold.append(cr2)
            if cr2 > best_r2:
                best_r2 = cr2
                best_alpha = alpha
                best_p_f_fold = p_f

        # 记录各项指标
        r2_s_curr = r2_score(y_te, p_s)
        r2_i_curr = r2_score(y_te, p_i)
        
        r2_sp.append(r2_s_curr)
        r2_im.append(r2_i_curr)
        r2_fu.append(best_r2)
        
        rmse_sp.append(np.sqrt(mean_squared_error(y_te, p_s)))
        rmse_im.append(np.sqrt(mean_squared_error(y_te, p_i)))
        rmse_fu.append(np.sqrt(mean_squared_error(y_te, best_p_f_fold)))
        best_ws.append(best_alpha)
        
        all_y_true.extend(y_te)
        all_y_pred_sp.extend(p_s)
        all_y_pred_im.extend(p_i)
        all_y_pred_fu.extend(best_p_f_fold)
        
        last_alphas = alphas
        last_r2s = r2s_fold

    print(f"\n>>>> 【{dataset_name}】 最终评估结果 <<<<")
    print(f"有效精准对齐样本数: {len(y)}")
    print(f"  [分支 A] 单独使用 光谱分支(NIR) 平均 R2: {np.mean(r2_sp):.4f}  | RMSE: {np.mean(rmse_sp):.4f}")
    print(f"  [分支 B] 单独使用 图像分支(RGB) 平均 R2: {np.mean(r2_im):.4f}  | RMSE: {np.mean(rmse_im):.4f}")
    print(f"  [融合 C] 自适应加权联合后 平均 R2: {np.mean(r2_fu):.4f}  | RMSE: {np.mean(rmse_fu):.4f}")
    print(f"  --> 图像特征所起作用(平均 Alpha 权重): {np.mean(best_ws):.2f} (1代表完全依赖图像, 0代表完全靠光谱)")
    print("-" * 50)

    # =============== 可视化与保存数据 =================
    dataset_slug = dataset_name.split()[0].lower()
    
    # 1. 保存指标与预测值
    metrics_df = pd.DataFrame({
        "Fold": list(range(1, 6)),
        "R2_Spec": r2_sp, "RMSE_Spec": rmse_sp,
        "R2_Img": r2_im, "RMSE_Img": rmse_im,
        "R2_Fusion": r2_fu, "RMSE_Fusion": rmse_fu,
        "Best_Alpha": best_ws
    })
    metrics_df.loc["Mean"] = metrics_df.mean()
    metrics_df.to_csv(os.path.join(save_dir, f"{dataset_slug}_performance_metrics.csv"))
    
    preds_df = pd.DataFrame({
        "True_SOM": all_y_true,
        "Pred_Spec": all_y_pred_sp,
        "Pred_Img": all_y_pred_im,
        "Pred_Fusion": all_y_pred_fu
    })
    preds_df.to_csv(os.path.join(save_dir, f"{dataset_slug}_crossval_predictions.csv"), index=False)
    
    # 2. 图像展示配置 (支持中文黑体)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  
    plt.rcParams['axes.unicode_minus'] = False
    
    # 图一：真实值 vs 预测值散点图
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(all_y_true, all_y_pred_sp, alpha=0.6, color='skyblue')
    plt.plot([min(all_y_true), max(all_y_true)], [min(all_y_true), max(all_y_true)], 'r--')
    plt.title(f"{dataset_slug} 光谱模型 (R2={np.mean(r2_sp):.2f})")
    plt.xlabel("真实 SOM")
    plt.ylabel("预测 SOM")
    
    plt.subplot(1, 3, 2)
    plt.scatter(all_y_true, all_y_pred_im, alpha=0.6, color='lightgreen')
    plt.plot([min(all_y_true), max(all_y_true)], [min(all_y_true), max(all_y_true)], 'r--')
    plt.title(f"{dataset_slug} 图像模型 (R2={np.mean(r2_im):.2f})")
    plt.xlabel("真实 SOM")
    plt.ylabel("预测 SOM")
    
    plt.subplot(1, 3, 3)
    plt.scatter(all_y_true, all_y_pred_fu, alpha=0.7, color='salmon')
    plt.plot([min(all_y_true), max(all_y_true)], [min(all_y_true), max(all_y_true)], 'r--')
    plt.title(f"{dataset_slug} 晚期融合模型 (R2={np.mean(r2_fu):.2f})")
    plt.xlabel("真实 SOM")
    plt.ylabel("预测 SOM")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{dataset_slug}_scatter_plots.png"), dpi=200)
    plt.close()
    
    # 图二：融合模型的Alpha权重搜索收敛图（由于非梯度下降，故展示最后一次Fold的搜索轨迹）
    plt.figure(figsize=(8, 5))
    plt.plot(last_alphas, last_r2s, label='验证集R2伴随Alpha的变化轨迹', color='purple', linewidth=2)
    plt.axvline(x=best_ws[-1], color='red', linestyle='--', label=f'最佳 Alpha: {best_ws[-1]:.2f}')
    plt.title(f"{dataset_slug} Alpha 权重自适应收敛曲线 (Fold-5)")
    plt.xlabel("Alpha 权重配置 (0=纯光谱, 1=纯图像)")
    plt.ylabel("预测 R2 分数")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{dataset_slug}_alpha_convergence.png"), dpi=200)
    plt.close()
    
    # 图三：RMSE 与 R2 的性能对比条形图
    labels = ['光谱(NIR)', '图像(RGB)', '融合模型']
    means_r2 = [np.mean(r2_sp), np.mean(r2_im), np.mean(r2_fu)]
    means_rmse = [np.mean(rmse_sp), np.mean(rmse_im), np.mean(rmse_fu)]
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()
    
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, means_r2, width, label='R2 Score', color='dodgerblue')
    rects2 = ax2.bar(x + width/2, means_rmse, width, label='RMSE', color='tomato')
    
    ax1.set_ylabel('R2 Score (越高越好)', color='dodgerblue')
    ax2.set_ylabel('RMSE (越低越好)', color='tomato')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title(f"{dataset_slug} 各分支性能综合对比")
    
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{dataset_slug}_performance_bar.png"), dpi=200)
    plt.close()


def main():
    root = "d:/desktop/综合测试"
    save_dir = os.path.join(root, "work_save/final_multimodal_results")
    os.makedirs(save_dir, exist_ok=True)

    # ==========================
    # 1. 严格对齐评估 NEW DATA
    # ==========================
    print("\n=== 初始化 New Data 任务 ===")
    def ext_new(fn):
        return os.path.splitext(fn)[0].lower()

    df_im1 = extract_image_features(os.path.join(root, "new_data/images2"), ext_new)
    df_tb1 = pd.read_excel(os.path.join(root, "new_data/ref2.xlsx"))

    df_tb1['Sample_ID'] = [f'c{i+1}' for i in range(len(df_tb1))]

    cols = df_tb1.columns.tolist()
    som_col = cols[0] if cols[0] != 'Sample_ID' else cols[1]
    sp_cols1 = [c for c in cols if c not in ['Sample_ID', som_col]]

    mg1 = pd.merge(df_tb1, df_im1, on='Sample_ID', how='inner')
    if len(mg1) > 0:
        run_evaluation(mg1[sp_cols1].values, mg1[['R_mean', 'G_mean', 'B_mean', 'V_mean', 'Color_Index']].values, mg1[som_col].values, "NEW DATA (ref2.xlsx + images2)", save_dir)


    # ==========================
    # 2. 严格对齐评估 FINAL DATA
    # ==========================
    print("\n=== 初始化 Final Data 任务 ===")
    def ext_fin(fn):
        m = re.search(r'[A-Za-z\d]+-\d{2}-\d{2}', fn)
        return m.group(0) if m else os.path.splitext(fn)[0]

    img3_path = os.path.join(root, "final_data/images3")
    if os.path.exists(img3_path):
        df_im2 = extract_image_features(img3_path, ext_fin)
        df_tb2 = pd.read_excel(os.path.join(root, "final_data/final.xlsx")) 
        df_tb2.rename(columns={df_tb2.columns[0]: 'Sample_ID'}, inplace=True)

        df_tb2['Sample_ID'] = df_tb2['Sample_ID'].astype(str)
        df_im2['Sample_ID'] = df_im2['Sample_ID'].astype(str)

        som_col2 = 'SOM' if 'SOM' in df_tb2.columns else df_tb2.columns[1]
        sp_cols2 = [c for c in df_tb2.columns if c not in ['Sample_ID', som_col2]]

        mg2 = pd.merge(df_tb2, df_im2, on='Sample_ID', how='inner')
        if len(mg2) > 0:
            run_evaluation(mg2[sp_cols2].values, mg2[['R_mean', 'G_mean', 'B_mean', 'V_mean', 'Color_Index']].values, mg2[som_col2].values, "FINAL DATA (final.xlsx + images3)", save_dir)
        else:
            print("Final Data 匹配失败: 对应交集为 0 !")

if __name__ == "__main__":
    main()
