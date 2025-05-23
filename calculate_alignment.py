import csv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 1) Collect significant features per ROI based on FREM vs Dummy
roi_significant_features = {}
roi_r2 = defaultdict(dict)

for filename in ["results/ridge_frem_scores_backwards.csv",
                 "results/ridge_frem_scores_forward.csv"]:
    with open(filename, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            roi = row["ROI_label"]
            feature = row["Feature"]
            frem_score = float(row["FREM_score"])
            dummy_score = float(row["Dummy_FREM_score"])
            if frem_score > dummy_score:
                roi_significant_features.setdefault(roi, set()).add(feature)
                roi_r2[roi][feature] = float(row["FREM_score"])

# 2) Collect significant features per layer based on p_value <= 0.2
layer_significant_features = {}
layer_r2 = defaultdict(dict)

with open("results/probing_results.csv", "r", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        layer = row["layer"]
        feature = row["feature"]
        p_value = float(row["p_value"])
        if p_value <= 0.2:
            layer_significant_features.setdefault(layer, set()).add(feature)
            layer_r2[layer][feature] = float(row["actual_mean_r2"])

# 3) Build alignment matrix: number of overlapping features for each ROI/layer
all_rois = sorted(roi_r2.keys())
all_layers = sorted(layer_r2.keys())

# 4) Gather all features mentioned in either ROI or layer data
all_features = sorted(
    set().union(*(roi_r2[roi].keys() for roi in all_rois))
    .union(*(layer_r2[layer].keys() for layer in all_layers))
)

"""# 5) Sanity check: Calculate the number of significant features
print("ROI/Layer Alignment Matrix (counts of significant features):")
print("\t" + "\t".join(all_layers))
for roi in all_rois:
    row_counts = [str(roi_layer_alignment[roi][layer]) for layer in all_layers]
    print(f"{roi}\t" + "\t".join(row_counts))"""

# 6) Build numeric vectors (R² values) for each ROI
roi_vectors = {}
for roi in all_rois:
    roi_vectors[roi] = np.array(
        [roi_r2[roi].get(feature, 0.0) for feature in all_features]
    ).reshape(1, -1)

# 7) Build numeric vectors (R² values) for each layer
layer_vectors = {}
for layer in all_layers:
    layer_vectors[layer] = np.array(
        [layer_r2[layer].get(feature, 0.0) for feature in all_features]
    ).reshape(1, -1)

# 8) Calculate cosine similarity between each ROI and each layer
roi_layer_cosine = {}
for roi in all_rois:
    roi_layer_cosine[roi] = {}
    for layer in all_layers:
        sim = cosine_similarity(roi_vectors[roi], layer_vectors[layer])[0, 0]
        roi_layer_cosine[roi][layer] = sim

# 9) Example: Print a table of similarities
print("ROI-Layer Cosine Similarities (R²-based):")
print("\t" + "\t".join(all_layers))
for roi in all_rois:
    row_sims = [f"{roi_layer_cosine[roi][layer]:.3f}" for layer in all_layers]
    print(f"{roi}\t" + "\t".join(row_sims))

roi_names = sorted(roi_layer_cosine.keys())
layer_names = sorted(next(iter(roi_layer_cosine.values())).keys())

# 10) Plot the cosine similarity matrix
layers = ['layer_1_embeddings', 'layer_2_embeddings', 'layer_3_embeddings', 'layer_4_embeddings', 'layer_5_embeddings', 'layer_6_embeddings', 'layer_1_attention', 'layer_2_attention', 'layer_3_attention', 'layer_4_attention', 'layer_5_attention', 'layer_6_attention']

# Define a custom dictionary mapping each ROI label to a meaningful name
custom_roi_labels = {
    "2301": "IFG, opercularis (L)",
    "2302": "IFG, opercularis (R)",
    "2311": "IFG, triangularis (L)",
    "2312": "IFG, triangularis (R)",
    "2321": "IFG, orbitalis (L)",
    "2322": "IFG, orbitalis (R)",
    "6211": "SG (L)",
    "6212": "SG (R)",
    "6221": "AG (L)",
    "6222": "AG (R)",
    "8111": "STG (L)",
    "8112": "STG (R)"
}

# In your plotting code, replace the numeric ROI labels with names from custom_roi_labels
roi_display_names = [custom_roi_labels.get(roi, roi) for roi in roi_names]


matrix_data = []
for roi in roi_names:
    row = [roi_layer_cosine[roi][layer] for layer in layers]
    matrix_data.append(row)
matrix_data = np.array(matrix_data)

plt.figure(figsize=(8, 6))
im = plt.imshow(matrix_data, cmap="viridis", aspect="auto")
plt.colorbar(label="Cosine Similarity")
plt.xticks(range(len(layers)), layers, rotation=45, ha="right")
plt.yticks(range(len(roi_names)), roi_display_names)

# Annotate each tile with its value
for i in range(len(roi_names)):
    for j in range(len(layers)):
        plt.text(j, i, f"{matrix_data[i, j]:.2f}",
                 ha="center", va="center", color="white")

plt.title("ROI-Layer Cosine Similarity")
plt.tight_layout()
#plt.show()

# Separate left vs. right ROIs 
left_rois = [roi for roi in roi_names if roi.endswith("1")]
right_rois = [roi for roi in roi_names if roi.endswith("2")]

layers = [
    "layer_1_embeddings", "layer_2_embeddings", "layer_3_embeddings",
    "layer_4_embeddings", "layer_5_embeddings", "layer_6_embeddings",
    "layer_1_attention", "layer_2_attention", "layer_3_attention",
    "layer_4_attention", "layer_5_attention", "layer_6_attention"
]

custom_roi_labels_left = {
    "2301": "IFG, opercularis (L)",
    "2311": "IFG, triangularis (L)",
    "2321": "IFG, orbitalis (L)",
    "6211": "SG (L)",
    "6221": "AG (L)",
    "8111": "STG (L)"
}
custom_roi_labels_right = {
    "2302": "IFG, opercularis (R)",
    "2312": "IFG, triangularis (R)",
    "2322": "IFG, orbitalis (R)",
    "6212": "SG (R)",
    "6222": "AG (R)",
    "8112": "STG (R)"
}

roi_display_names_left = [custom_roi_labels_left.get(roi, roi) for roi in left_rois]
roi_display_names_right = [custom_roi_labels_right.get(roi, roi) for roi in right_rois]

# 11) Show plots for each hempisphere separately
# Plot left hemisphere
plt.figure(figsize=(8, 6))
left_matrix_data = [
    [roi_layer_cosine[roi][layer] for layer in layers] 
    for roi in left_rois
]
left_matrix_data = np.array(left_matrix_data)
plt.imshow(left_matrix_data, cmap="viridis", aspect="auto", vmax=0.36)
plt.colorbar(label="Cosine Similarity")
plt.xticks(range(len(layers)), layers, rotation=45, ha="right")
plt.yticks(range(len(left_rois)), roi_display_names_left)

# Annotate values
for i in range(len(left_rois)):
    for j in range(len(layers)):
        plt.text(j, i, f"{left_matrix_data[i, j]:.2f}",
                 ha="center", va="center", color="white")

plt.title("Left Hemisphere ROI-Layer Cosine Similarity")
plt.tight_layout()
#plt.show()

# Plot right hemisphere
plt.figure(figsize=(8, 6))
right_matrix_data = [
    [roi_layer_cosine[roi][layer] for layer in layers] 
    for roi in right_rois
]
right_matrix_data = np.array(right_matrix_data)
plt.imshow(right_matrix_data, cmap="viridis", aspect="auto", vmax=0.36)
plt.colorbar(label="Cosine Similarity")
plt.xticks(range(len(layers)), layers, rotation=45, ha="right")
plt.yticks(range(len(right_rois)), roi_display_names_right)

# Annotate values
for i in range(len(right_rois)):
    for j in range(len(layers)):
        plt.text(j, i, f"{right_matrix_data[i, j]:.2f}",
                 ha="center", va="center", color="white")

plt.title("Right Hemisphere ROI-Layer Cosine Similarity")
plt.tight_layout()
#plt.show()

# 11) Calculate feature-wise alignment

feature_similarity = {}

for feature in all_features:
    # Initialize an empty dict before assigning keys
    if feature not in feature_similarity:
        feature_similarity[feature] = {}

    roi_vals = np.array([roi_r2[roi].get(feature, 0.0) for roi in all_rois])
    layer_vals = np.array([layer_r2[layer].get(feature, 0.0) for layer in all_layers])
    
    roi_2d = roi_vals.reshape(1, -1)
    layer_2d = layer_vals.reshape(1, -1)
    
    if np.linalg.norm(roi_vals) == 0 or np.linalg.norm(layer_vals) == 0:
        feature_similarity[feature]["cosine"] = 0.0
    else:
        cos_val = cosine_similarity(roi_2d, layer_2d)[0, 0]
        feature_similarity[feature]["cosine"] = cos_val

# Example: print results
for feat, vals in feature_similarity.items():
    print(f"{feat}: Cosine Similarity = {vals.get('cosine', 0.0):.3f}")


# You can also visualize this data if needed
# For example, you can create a bar plot of the ratios
"""plt.figure(figsize=(10, 6))
features = list(feature_similarity.keys())
cosines = [feature_similarity[feature]["cosine"] for feature in features]
plt.barh(features, cosines, color='skyblue')
plt.xlabel("Ratio of ROI to Layer R²")
plt.title("Feature-wise Alignment cosines")
plt.axvline(x=1, color='red', linestyle='--', label='Equal Alignment')
plt.legend()
plt.tight_layout()
plt.show()"""

# 12 Analyse difference between discourse-level and syntactic features

syntactic_features = ['avg_sentence_length', 'avg_,', 'avg_.', 'avg_:', 'avg_ADV', 'avg_AMOD', 'avg_CC', 'avg_CC.1', 'avg_CD', 'avg_COORD', 'avg_DEP', 'avg_DT', 'avg_IN', 'avg_IOBJ', 'avg_JJ', 'avg_MD', 'avg_NMOD', 'avg_NN', 'avg_NNP', 'avg_NNS', 'avg_OBJ', 'avg_P', 'avg_PMOD', 'avg_POS', 'avg_PRD', 'avg_PRN', 'avg_PRP', 'avg_PRP$', 'avg_PRT', 'avg_RB', 'avg_ROOT', 'avg_RP', 'avg_SBJ', 'avg_TO', 'avg_UH', 'avg_VB', 'avg_VBD', 'avg_VBG', 'avg_VBN', 'avg_VBP', 'avg_VBZ', 'avg_VC', 'avg_VMOD', 'avg_WDT', 'avg_WP', 'avg_WRB']
discourse_features = ['avg_annoyed', 'avg_annoyed_sticky', 'avg_be', 'avg_collidePhys_sticky', 'avg_commanding', 'avg_commanding_sticky', 'avg_cynical_sticky', 'avg_dislike', 'avg_dislike_sticky', 'avg_draco', 'avg_fear', 'avg_fear_sticky', 'avg_filch', 'avg_fly', 'avg_fly_sticky', 'avg_harry', 'avg_hear', 'avg_herm', 'avg_hooch', 'avg_hurtMental_sticky', 'avg_hurtPhys_sticky', 'avg_know', 'avg_likavg_like_sticky', 'avg_manipulate_sticky', 'avg_minerva', 'avg_move_sticky', 'avg_nervousness', 'avg_nervousness_sticky', 'avg_neville', 'avg_peeves', 'avg_pleading_sticky', 'avg_praising_sticky', 'avg_pride_sticky', 'avg_questioning', 'avg_questioning_sticky', 'avg_relief_sticky', 'avg_ron', 'avg_see', 'avg_speak', 'avg_speak_sticky', 'avg_tell', 'avg_var_WL', 'avg_wonder', 'avg_wonder_sticky', 'avg_wood', 'avg_word_length']

print("Discourse Features:")
for feature in discourse_features:
    print(feature)

print("\nSyntactic Features:")
for feature in syntactic_features:
    print(feature)

# 13) Per layer: average R² for discourse features and syntactic features
layer_discourse_r2 = {}
layer_syntactic_r2 = {}
for layer in all_layers:
    # Collect only significant features for this layer
    sig_feats = layer_significant_features.get(layer, set())
    
    # Discourse features intersection
    discourse_feats = sig_feats.intersection(discourse_features)
    # Syntactic features intersection
    syntactic_feats = sig_feats.intersection(syntactic_features)
    
    # Calculate mean R² for discourse
    if len(discourse_feats) > 0:
        discourse_vals = [layer_r2[layer][f] for f in discourse_feats]
        layer_discourse_r2[layer] = np.mean(discourse_vals)
    else:
        layer_discourse_r2[layer] = 0.0
    
    # Calculate mean R² for syntactic
    if len(syntactic_feats) > 0:
        syntactic_vals = [layer_r2[layer][f] for f in syntactic_feats]
        layer_syntactic_r2[layer] = np.mean(syntactic_vals)
    else:
        layer_syntactic_r2[layer] = 0.0

# 14) Per ROI: average R² for discourse features and syntactic features
roi_discourse_r2 = {}
roi_syntactic_r2 = {}
for roi in all_rois:
    # Collect only significant features for this ROI
    sig_feats = roi_significant_features.get(roi, set())
    
    # Discourse features intersection
    discourse_feats = sig_feats.intersection(discourse_features)
    # Syntactic features intersection
    syntactic_feats = sig_feats.intersection(syntactic_features)
    
    # Calculate mean R² for discourse
    if len(discourse_feats) > 0:
        discourse_vals = [roi_r2[roi][f] for f in discourse_feats]
        roi_discourse_r2[roi] = np.mean(discourse_vals)
    else:
        roi_discourse_r2[roi] = 0.0
    
    # Calculate mean R² for syntactic
    if len(syntactic_feats) > 0:
        syntactic_vals = [roi_r2[roi][f] for f in syntactic_feats]
        roi_syntactic_r2[roi] = np.mean(syntactic_vals)
    else:
        roi_syntactic_r2[roi] = 0.0

print("=== Layer-Level Averages ===")
for layer in all_layers:
    print(f"{layer} - Discourse R²: {layer_discourse_r2[layer]:.4f}, "
          f"Syntactic R²: {layer_syntactic_r2[layer]:.4f}")

print("\n=== ROI-Level Averages ===")
for roi in all_rois:
    print(f"{roi} - Discourse R²: {roi_discourse_r2[roi]:.4f}, "
          f"Syntactic R²: {roi_syntactic_r2[roi]:.4f}")
    

# 1) Plot layer-level averages in a grouped bar chart
layer_labels = layers
layer_discourse_vals = [layer_discourse_r2[l] for l in layer_labels]
layer_syntactic_vals = [layer_syntactic_r2[l] for l in layer_labels]

x = np.arange(len(layer_labels))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, layer_discourse_vals, width, label='Discourse Avg R²')
plt.bar(x + width/2, layer_syntactic_vals, width, label='Syntactic Avg R²')
plt.xticks(x, layer_labels, rotation=45, ha='right')
plt.ylabel('Average R²')
plt.title('Layer-Level Average R² (Discourse vs. Syntactic)')
plt.legend()
plt.tight_layout()
#plt.show()

# 2) Plot ROI-level averages in a grouped bar chart
roi_labels = all_rois
roi_discourse_vals = [roi_discourse_r2[r] for r in roi_labels]
roi_syntactic_vals = [roi_syntactic_r2[r] for r in roi_labels]

x = np.arange(len(roi_labels))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, roi_discourse_vals, width, label='Discourse Avg R²')
plt.bar(x + width/2, roi_syntactic_vals, width, label='Syntactic Avg R²')
plt.xticks(x, roi_display_names, rotation=45, ha='right')
plt.ylabel('Average R²')
plt.title('ROI-Level Average R² (Discourse vs. Syntactic)')
plt.legend()
plt.tight_layout()
#plt.show()

## 1) Discourse-based ROI/Layer vectors
roi_discourse_vectors = {}
for roi in all_rois:
    vals = [roi_r2[roi].get(f, 0.0) for f in discourse_features]
    roi_discourse_vectors[roi] = np.array(vals).reshape(1, -1)

layer_discourse_vectors = {}
for layer in all_layers:
    vals = [layer_r2[layer].get(f, 0.0) for f in discourse_features]
    layer_discourse_vectors[layer] = np.array(vals).reshape(1, -1)

# Calculate discourse-only cosine similarity
discourse_roi_layer_cosine = {}
for roi in all_rois:
    discourse_roi_layer_cosine[roi] = {}
    for layer in all_layers:
        # If either vector is all zeros => cos similarity = 0
        if (np.linalg.norm(roi_discourse_vectors[roi]) == 0 or
            np.linalg.norm(layer_discourse_vectors[layer]) == 0):
            discourse_roi_layer_cosine[roi][layer] = 0.0
        else:
            sim = cosine_similarity(
                roi_discourse_vectors[roi],
                layer_discourse_vectors[layer]
            )[0, 0]
            discourse_roi_layer_cosine[roi][layer] = sim

## 2) Syntactic-based ROI/Layer vectors
roi_syntactic_vectors = {}
for roi in all_rois:
    vals = [roi_r2[roi].get(f, 0.0) for f in syntactic_features]
    roi_syntactic_vectors[roi] = np.array(vals).reshape(1, -1)

layer_syntactic_vectors = {}
for layer in all_layers:
    vals = [layer_r2[layer].get(f, 0.0) for f in syntactic_features]
    layer_syntactic_vectors[layer] = np.array(vals).reshape(1, -1)

# Calculate syntactic-only cosine similarity
syntactic_roi_layer_cosine = {}
for roi in all_rois:
    syntactic_roi_layer_cosine[roi] = {}
    for layer in all_layers:
        if (np.linalg.norm(roi_syntactic_vectors[roi]) == 0 or
            np.linalg.norm(layer_syntactic_vectors[layer]) == 0):
            syntactic_roi_layer_cosine[roi][layer] = 0.0
        else:
            sim = cosine_similarity(
                roi_syntactic_vectors[roi],
                layer_syntactic_vectors[layer]
            )[0, 0]
            syntactic_roi_layer_cosine[roi][layer] = sim

# Example: Print out for each ROI/layer
print("=== Discourse-Only Cosine Similarities ===")
print("\t" + "\t".join(all_layers))
for roi in all_rois:
    row_sims = [f"{discourse_roi_layer_cosine[roi][layer]:.3f}" for layer in all_layers]
    print(f"{roi}\t" + "\t".join(row_sims))

print("\n=== Syntactic-Only Cosine Similarities ===")
print("\t" + "\t".join(all_layers))
for roi in all_rois:
    row_sims = [f"{syntactic_roi_layer_cosine[roi][layer]:.3f}" for layer in all_layers]
    print(f"{roi}\t" + "\t".join(row_sims))

# Visualize the discourse-only cosine similarities
discourse_matrix_data = []
for roi in all_rois:
    row = [discourse_roi_layer_cosine[roi][layer] for layer in layers]
    discourse_matrix_data.append(row)
discourse_matrix_data = np.array(discourse_matrix_data)

plt.figure(figsize=(8, 6))
im = plt.imshow(discourse_matrix_data, cmap="viridis", aspect="auto")
plt.colorbar(label="Cosine Similarity (Discourse)")
plt.xticks(range(len(all_layers)), layers, rotation=45, ha="right")
plt.yticks(range(len(all_rois)), roi_display_names)

# Annotate each tile with its value
for i in range(len(all_rois)):
    for j in range(len(all_layers)):
        plt.text(j, i, f"{discourse_matrix_data[i, j]:.2f}",
                 ha="center", va="center", color="white")

plt.title("ROI-Layer Cosine Similarity (Discourse Features)")
plt.tight_layout()

# Visualize the syntactic-only cosine similarities
syntactic_matrix_data = []
for roi in all_rois:
    row = [syntactic_roi_layer_cosine[roi][layer] for layer in layers]
    syntactic_matrix_data.append(row)
syntactic_matrix_data = np.array(syntactic_matrix_data)

plt.figure(figsize=(8, 6))
im = plt.imshow(syntactic_matrix_data, cmap="viridis", aspect="auto")
plt.colorbar(label="Cosine Similarity (Syntactic)")
plt.xticks(range(len(all_layers)), layers, rotation=45, ha="right")
plt.yticks(range(len(all_rois)), roi_display_names)

# Annotate each tile with its value
for i in range(len(all_rois)):
    for j in range(len(all_layers)):
        plt.text(j, i, f"{syntactic_matrix_data[i, j]:.2f}",
                 ha="center", va="center", color="white")

plt.title("ROI-Layer Cosine Similarity (Syntactic Features)")
plt.tight_layout()
plt.show()
# ...existing code.