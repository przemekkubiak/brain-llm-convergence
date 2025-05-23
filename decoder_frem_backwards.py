import numpy as np
from nilearn.decoding import FREMRegressor
from nilearn.image import load_img, iter_img, new_img_like, resample_to_img
from nilearn.plotting import plot_glass_brain, plot_stat_map, show
from nilearn.maskers import NiftiMasker
from nilearn.datasets import fetch_atlas_aal
from sklearn.preprocessing import StandardScaler
import csv

d = np.load("/mnt/nas_home/pk571/source-brains_encoder/data/aligned_data_subject_2_embeddings_features.npz", allow_pickle=True)
#d = np.load("source-brains_encoder/data/aligned_data_subject_2_embeddings_features.npz", allow_pickle=True)
d = d["aligned_data"]

# Define the list of features
features = ['avg_speak_sticky', 'avg_speak', 'avg_fly_sticky', 'avg_manipulate_sticky', 'avg_move_sticky', 'avg_collidePhys_sticky', 'avg_fly', 'avg_manipulate', 'avg_move', 'avg_annoyed', 'avg_commanding', 'avg_dislike', 'avg_fear', 'avg_like', 'avg_nervousness', 'avg_questioning', 'avg_wonder', 'avg_annoyed_sticky', 'avg_commanding_sticky', 'avg_cynical_sticky', 'avg_dislike_sticky', 'avg_fear_sticky', 'avg_hurtMental_sticky', 'avg_hurtPhys_sticky', 'avg_like_sticky', 'avg_nervousness_sticky', 'avg_pleading_sticky', 'avg_praising_sticky', 'avg_pride_sticky', 'avg_questioning_sticky', 'avg_relief_sticky', 'avg_wonder_sticky', 'avg_be', 'avg_hear', 'avg_know', 'avg_see', 'avg_tell', 'avg_draco', 'avg_filch', 'avg_harry', 'avg_herm', 'avg_hooch', 'avg_minerva', 'avg_neville', 'avg_peeves', 'avg_ron', 'avg_wood', 'avg_word_length', 'avg_var_WL', 'avg_sentence_length', 'avg_,', 'avg_.', 'avg_:', 'avg_CC', 'avg_CD', 'avg_DT', 'avg_IN', 'avg_JJ', 'avg_MD', 'avg_NN', 'avg_NNP', 'avg_NNS', 'avg_POS', 'avg_PRP', 'avg_PRP$', 'avg_RB', 'avg_RP', 'avg_TO', 'avg_UH', 'avg_VB', 'avg_VBD', 'avg_VBG', 'avg_VBN', 'avg_VBP', 'avg_VBZ', 'avg_WDT', 'avg_WP', 'avg_WRB', 'avg_ADV', 'avg_AMOD', 'avg_CC.1', 'avg_COORD', 'avg_DEP', 'avg_IOBJ', 'avg_NMOD', 'avg_OBJ', 'avg_P', 'avg_PMOD', 'avg_PRD', 'avg_PRN', 'avg_PRT', 'avg_ROOT', 'avg_SBJ', 'avg_VC', 'avg_VMOD']

# Load preprocessed fMRI data
fmri_img = load_img("/mnt/nas_home/pk571/source-brains_encoder/data//subject_2_preprocessed_volume.nii")
#fmri_img = load_img("source-brains_encoder/data//subject_2_preprocessed_volume.nii")

# Load the brain mask
#mask_img = load_img("source-brains_encoder/data/subject_2_preprocessed_volume_mask.nii")

#masker = NiftiMasker(mask_img=mask_img, standardize="zscore_sample")

volumes = list(iter_img(fmri_img))  # Each volume is now a 3D Niimg-like object

# Load the atlas
atlas = fetch_atlas_aal()
atlas_img = load_img(atlas["maps"])
atlas_data = atlas_img.get_fdata()

# Define the list of roi labels
roi_labels = ['8112', '8111', '6222', '6221', '6212', '6211'] 

scores_dict = {}

# Iterate through each region
for roi_label in roi_labels:
    print(f'Processing ROI: {roi_label}')
    roi_label = int(roi_label)
    roi_data = atlas_data == roi_label
    roi_mask_img = new_img_like(atlas_img, roi_data)

    resampled_mask_img = resample_to_img(roi_mask_img, fmri_img, interpolation='nearest')

    roi_masker = NiftiMasker(mask_img=resampled_mask_img, standardize="zscore_sample")

    # Initialize the FREMRegressor
    frem = FREMRegressor("ridge", cv=5, standardize="zscore_sample", mask=roi_masker, clustering_percentile=30, screening_percentile=4, scoring='r2')
    
    # Create a baseline FREM model using a dummy regressor
    dummy_frem = FREMRegressor("dummy_regressor", cv=5, standardize="zscore_sample", mask=roi_masker, clustering_percentile=30, screening_percentile=40, scoring='r2')

    # Gather the feature data
    for feature in features:
        print(f'Processing feature: {feature}')

        X_single = np.array([row[feature] for row in d])

        scaler = StandardScaler()

        X_single = scaler.fit_transform(X_single.reshape(-1, 1)).ravel()

        print(X_single)

        # Fit FREM model
        frem.fit(volumes, X_single)
        frem_scores = frem.score(volumes, X_single)
        print("FREM scores:", frem_scores)

        # Fit dummy baseline
        dummy_frem.fit(volumes, X_single)
        dummy_frem_scores = dummy_frem.score(volumes, X_single)
        print("Dummy-FREM scores:", dummy_frem_scores)

        scores_dict.setdefault(roi_label, {})[feature] = {
            "frem_score": frem_scores,
            "dummy_score": dummy_frem_scores
        }

# Save to CSV
with open("ridge_frem_scores_backwards.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ROI_label", "Feature", "FREM_score", "Dummy_FREM_score"])
    for roi_label, features_data in scores_dict.items():
        for feature, scores_data in features_data.items():
            writer.writerow([
                roi_label,
                feature,
                scores_data["frem_score"],
                scores_data["dummy_score"]
            ])
        
