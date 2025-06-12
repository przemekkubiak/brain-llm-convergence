import numpy as np
import csv
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import os
from torch.utils.data import DataLoader, TensorDataset


#/mnt/nas_home/pk571/

# Load data
data = np.load("/mnt/nas_home/pk571/source-brains_encoder/data/aligned_data_subject_2_embeddings_features.npz", allow_pickle=True)
d = data["aligned_data"]

# Define linguistic features and layers
#features_names = ('avg_s1000_1', 'avg_s1000_10', 'avg_s1000_15', 'avg_s1000_21', 'avg_s1000_30', 'avg_s1000_48', 'avg_s1000_66', 'avg_s1000_72', 'avg_s1000_75', 'avg_s1000_82', 'avg_s1000_97', 'avg_s1000_98', 'avg_s1000_117', 'avg_s1000_132', 'avg_s1000_135', 'avg_s1000_163', 'avg_s1000_170', 'avg_s1000_182', 'avg_s1000_197', 'avg_s1000_213', 'avg_s1000_222', 'avg_s1000_253', 'avg_s1000_255', 'avg_s1000_264', 'avg_s1000_282', 'avg_s1000_284', 'avg_s1000_286', 'avg_s1000_297', 'avg_s1000_298', 'avg_s1000_303', 'avg_s1000_312', 'avg_s1000_319', 'avg_s1000_323', 'avg_s1000_324', 'avg_s1000_334', 'avg_s1000_349', 'avg_s1000_373', 'avg_s1000_380', 'avg_s1000_389', 'avg_s1000_399', 'avg_s1000_430', 'avg_s1000_435', 'avg_s1000_447', 'avg_s1000_454', 'avg_s1000_456', 'avg_s1000_467', 'avg_s1000_469', 'avg_s1000_474', 'avg_s1000_475', 'avg_s1000_488', 'avg_s1000_493', 'avg_s1000_509', 'avg_s1000_512', 'avg_s1000_519', 'avg_s1000_527', 'avg_s1000_535', 'avg_s1000_538', 'avg_s1000_545', 'avg_s1000_549', 'avg_s1000_557', 'avg_s1000_560', 'avg_s1000_566', 'avg_s1000_582', 'avg_s1000_586', 'avg_s1000_592', 'avg_s1000_603', 'avg_s1000_612', 'avg_s1000_615', 'avg_s1000_620', 'avg_s1000_635', 'avg_s1000_646', 'avg_s1000_653', 'avg_s1000_673', 'avg_s1000_681', 'avg_s1000_696', 'avg_s1000_698', 'avg_s1000_700', 'avg_s1000_722', 'avg_s1000_724', 'avg_s1000_774', 'avg_s1000_775', 'avg_s1000_777', 'avg_s1000_779', 'avg_s1000_780', 'avg_s1000_784', 'avg_s1000_796', 'avg_s1000_797', 'avg_s1000_808', 'avg_s1000_856', 'avg_s1000_859', 'avg_s1000_864', 'avg_s1000_866', 'avg_s1000_873', 'avg_s1000_893', 'avg_s1000_928', 'avg_s1000_946', 'avg_s1000_971', 'avg_s1000_977', 'avg_s1000_978', 'avg_s1000_984', 'avg_speak_sticky', 'avg_speak', 'avg_fly_sticky', 'avg_manipulate_sticky', 'avg_move_sticky', 'avg_collidePhys_sticky', 'avg_fly', 'avg_manipulate', 'avg_move', 'avg_annoyed', 'avg_commanding', 'avg_dislike', 'avg_fear', 'avg_like', 'avg_nervousness', 'avg_questioning', 'avg_wonder', 'avg_annoyed_sticky', 'avg_commanding_sticky', 'avg_cynical_sticky', 'avg_dislike_sticky', 'avg_fear_sticky', 'avg_hurtMental_sticky', 'avg_hurtPhys_sticky', 'avg_like_sticky', 'avg_nervousness_sticky', 'avg_pleading_sticky', 'avg_praising_sticky', 'avg_pride_sticky', 'avg_questioning_sticky', 'avg_relief_sticky', 'avg_wonder_sticky', 'avg_be', 'avg_hear', 'avg_know', 'avg_see', 'avg_tell', 'avg_draco', 'avg_filch', 'avg_harry', 'avg_herm', 'avg_hooch', 'avg_minerva', 'avg_neville', 'avg_peeves', 'avg_ron', 'avg_wood', 'avg_word_length', 'avg_var_WL', 'avg_sentence_length', 'avg_,', 'avg_.', 'avg_:', 'avg_CC', 'avg_CD', 'avg_DT', 'avg_IN', 'avg_JJ', 'avg_MD', 'avg_NN', 'avg_NNP', 'avg_NNS', 'avg_POS', 'avg_PRP', 'avg_PRP$', 'avg_RB', 'avg_RP', 'avg_TO', 'avg_UH', 'avg_VB', 'avg_VBD', 'avg_VBG', 'avg_VBN', 'avg_VBP', 'avg_VBZ', 'avg_WDT', 'avg_WP', 'avg_WRB', 'avg_ADV', 'avg_AMOD', 'avg_CC.1', 'avg_COORD', 'avg_DEP', 'avg_IOBJ', 'avg_NMOD', 'avg_OBJ', 'avg_P', 'avg_PMOD', 'avg_PRD', 'avg_PRN', 'avg_PRT', 'avg_ROOT', 'avg_SBJ', 'avg_VC', 'avg_VMOD')

features_names = ('avg_speak_sticky', 'avg_speak', 'avg_fly_sticky', 'avg_manipulate_sticky', 'avg_move_sticky', 'avg_collidePhys_sticky', 'avg_fly', 'avg_manipulate', 'avg_move', 'avg_annoyed', 'avg_commanding', 'avg_dislike', 'avg_fear', 'avg_like', 'avg_nervousness', 'avg_questioning', 'avg_wonder', 'avg_annoyed_sticky', 'avg_commanding_sticky', 'avg_cynical_sticky', 'avg_dislike_sticky', 'avg_fear_sticky', 'avg_hurtMental_sticky', 'avg_hurtPhys_sticky', 'avg_like_sticky', 'avg_nervousness_sticky', 'avg_pleading_sticky', 'avg_praising_sticky', 'avg_pride_sticky', 'avg_questioning_sticky', 'avg_relief_sticky', 'avg_wonder_sticky', 'avg_be', 'avg_hear', 'avg_know', 'avg_see', 'avg_tell', 'avg_draco', 'avg_filch', 'avg_harry', 'avg_herm', 'avg_hooch', 'avg_minerva', 'avg_neville', 'avg_peeves', 'avg_ron', 'avg_wood', 'avg_word_length', 'avg_var_WL', 'avg_sentence_length', 'avg_,', 'avg_.', 'avg_:', 'avg_CC', 'avg_CD', 'avg_DT', 'avg_IN', 'avg_JJ', 'avg_MD', 'avg_NN', 'avg_NNP', 'avg_NNS', 'avg_POS', 'avg_PRP', 'avg_PRP$', 'avg_RB', 'avg_RP', 'avg_TO', 'avg_UH', 'avg_VB', 'avg_VBD', 'avg_VBG', 'avg_VBN', 'avg_VBP', 'avg_VBZ', 'avg_WDT', 'avg_WP', 'avg_WRB', 'avg_ADV', 'avg_AMOD', 'avg_CC.1', 'avg_COORD', 'avg_DEP', 'avg_IOBJ', 'avg_NMOD', 'avg_OBJ', 'avg_P', 'avg_PMOD', 'avg_PRD', 'avg_PRN', 'avg_PRT', 'avg_ROOT', 'avg_SBJ', 'avg_VC', 'avg_VMOD')

# Example: probing embeddings and attention at different layers
layers = [f"layer_{i}_embeddings" for i in range(1, 7)] + [f"layer_{i}_attention" for i in range(1, 7)]

# Initialize model
class RidgeRegression(nn.Module):
    def __init__(self, input_dim, alpha=1.0):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.alpha = alpha

    def forward(self, x):
        return self.linear(x)

    def l2_penalty(self):
        return self.alpha * torch.sum(self.linear.weight ** 2)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=50):
    for epoch in range(num_epochs):
        model.train()
        for X_train, y_train in train_loader:
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train) + model.module.l2_penalty()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                val_outputs = model(X_val)
                val_loss += criterion(val_outputs, y_val) + model.module.l2_penalty()
        val_loss /= len(val_loader)
        if epoch in (0, 49, 99):
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Val Loss: {val_loss.item()}')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize KFold
cv = KFold(n_splits=3, shuffle=True, random_state=42)

n_permutations = 10  # Number of permutations for significance testing

results = []

for layer in layers:
    X_layer = []
    for row in d:
        if isinstance(row[layer], np.ndarray):
            X_layer.append(row[layer].flatten())
        else:
            print(f"Non-array element found in layer {layer}: {row[layer]}")
    # Ensure all elements have the same shape
    X_layer = [x for x in X_layer if x.shape == X_layer[0].shape]
    X_layer = np.array(X_layer, dtype=np.float32)
    X_layer = torch.tensor(X_layer, dtype=torch.float32).to(device)
    
    for feature in features_names:
        y_feature = np.array([row[feature] for row in d])
        y_feature = torch.tensor(y_feature, dtype=torch.float32).view(-1, 1).to(device)
        
        # Collect only rows whose layer embeddings and feature arrays match the shape of the first valid element
        X_filtered = []
        y_filtered = []

        first_shape = None
        for row in d:
            val = row[layer]
            if isinstance(val, np.ndarray):
                if first_shape is None:
                    first_shape = val.shape  # record the first valid shape
                if val.shape == first_shape:
                    X_filtered.append(val.flatten())
                    y_filtered.append(row[feature])
                else:
                    print(f"Skipping row with shape {val.shape}, expected {first_shape}")
            else:
                print(f"Skipping non-array element: {val}")

        X_filtered = np.array(X_filtered, dtype=np.float32)
        X_layer = torch.tensor(X_filtered, dtype=torch.float32).to(device)
        y_feature = torch.tensor(y_filtered, dtype=torch.float32).view(-1, 1).to(device)

        # Create DataLoader for mini-batch gradient descent
        dataset = TensorDataset(X_layer, y_feature)
        train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=128, shuffle=False)

        baseline_scores = []
        for train_idx, val_idx in cv.split(X_layer):
            X_train, X_val = X_layer[train_idx], X_layer[val_idx]
            y_train, y_val = y_feature[train_idx], y_feature[val_idx]

            # ----- MEAN BASELINE -----
            # 1) Compute the mean of y_train
            y_train_mean = y_train.mean()

            # 2) Create baseline predictions for y_val using that mean
            y_val_pred = torch.full_like(y_val, y_train_mean.item())

            # 3) Compute MSE loss for baseline
            baseline_loss = nn.MSELoss()(y_val_pred, y_val)
            baseline_scores.append(baseline_loss.item())

        mean_baseline_score = np.mean(baseline_scores)
        
        # Calculate actual score
        actual_scores = []
        for train_idx, val_idx in cv.split(X_layer):
            X_train, X_val = X_layer[train_idx], X_layer[val_idx]
            y_train, y_val = y_feature[train_idx], y_feature[val_idx]

            model = RidgeRegression(input_dim=X_layer.shape[1], alpha=1.0).to(device)
            model = nn.DataParallel(model)  # Use DataParallel to utilize multiple GPUs
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=50)  # Reduce epochs for faster execution

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val) + model.module.l2_penalty()
                actual_scores.append(val_loss.item())

        actual_mean_score = np.mean(actual_scores)
        
        # Permutation testing
        perm_scores = []
        for _ in range(n_permutations):
            y_permuted = shuffle(y_feature.cpu().numpy())
            y_permuted = torch.tensor(y_permuted, dtype=torch.float32).view(-1, 1).to(device)
            perm_score = []
            for train_idx, val_idx in cv.split(X_layer):
                X_train, X_val = X_layer[train_idx], X_layer[val_idx]
                y_train, y_val = y_permuted[train_idx], y_permuted[val_idx]

                model = RidgeRegression(input_dim=X_layer.shape[1], alpha=1.0).to(device)
                model = nn.DataParallel(model)  # Use DataParallel to utilize multiple GPUs
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

                train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=50)  # Reduce epochs for faster execution

                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val)
                    val_loss = criterion(val_outputs, y_val) + model.module.l2_penalty()
                    perm_score.append(val_loss.item())

            perm_scores.append(np.mean(perm_score))
        
        # Calculate p-value
        p_value = np.sum(np.array(perm_scores) >= actual_mean_score) / n_permutations
        
        # Store results
        results.append({
            "layer": layer,
            "feature": feature,
            "actual_mean_r2": actual_mean_score,
            "baseline_score": mean_baseline_score,
            "p_value": p_value
        })


# Ensure the results directory exists
os.makedirs("results", exist_ok=True)

# Save results to a CSV file
with open("results/probing_results.csv", "w", newline='') as csvfile:
    fieldnames = ["layer", "feature", "actual_mean_r2", "p_value"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for result in results:
        writer.writerow(result)
