
#Data processing and model evaluation script
import os
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def wish(df):
    # Keep only 1-3 story buildings
    df = df[df['nStory'].isin([1, 2, 3])]
    #  One-hot encoding for structure type
    categories1 = ['C1', 'C2', 'C3', 'RM1', 'RM2', 'S1' , 'S2', 'S3', 'S4', 'S5','URM','W1', 'W2']
    df['strutype'] = pd.Categorical(df['strutype'], categories=categories1, ordered=False)
    df = pd.get_dummies(df, columns=['strutype'], drop_first=False)
    # Year type categorization
    df.loc[df['year'] > 1989, 'yeartype'] = 1
    df.loc[(df['year'] <= 1989) & (df['year'] > 1978), 'yeartype'] = 2
    df.loc[df['year'] <= 1978, 'yeartype'] = 3
    categories1 = [1, 2, 3]
    df['yeartype'] = pd.Categorical(df['yeartype'], categories=categories1, ordered=False)
    df = pd.get_dummies(df, columns=['yeartype'], drop_first=False)
    #Set Bracketed Duration to 0.0001 if 0
    df.loc[df['Bracketed Duration'] == 0, 'Bracketed Duration'] = 0.0001
    return df


# Student model structure
class StudentModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# Multi-head attention feature extractor
class MultiHeadAttentionFeatureExtractor(nn.Module):
    def __init__(self, input_dim, num_heads=2):
        super(MultiHeadAttentionFeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.input_projection = nn.Linear(input_dim, 24)  #Project to 24 dims
        # Define multi-head attention layer
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=24,  # Input dim
            num_heads=num_heads,  #  Number of heads
            batch_first=True      #  Batch first
        )
        self.feature_projection = nn.Sequential(
            nn.Linear(24, 20)
        )
    def forward(self, x):
        x = self.input_projection(x)
        # (batch_size, features) -> (batch_size, 1, features) | Reshape for attention
        x = x.unsqueeze(1)
        attn_output, _ = self.multihead_attn(x, x, x)
        x = attn_output.squeeze(1)
        x = self.feature_projection(x)
        return x

# Multi-head attention feature extractor 2
class MultiHeadAttentionFeatureExtractor1(nn.Module):
    def __init__(self, input_dim, num_heads=2):
        super(MultiHeadAttentionFeatureExtractor1, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.input_projection = nn.Linear(input_dim, 24)  # Project to 24 dims
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=24,  # Input dim
            num_heads=num_heads,  #  Number of heads
            batch_first=True      #  Batch first
        )
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(24, 20)
        )
    def forward(self, x):
        x = self.input_projection(x) 
        x = x.unsqueeze(1)
        attn_output, _ = self.multihead_attn(x, x, x)
        x = attn_output.squeeze(1)
        x = self.feature_projection(x)
        return x

#  Deep Neural Network
class DNN(nn.Module):
    def __init__(self, input_dim,Drop=0.1):
        super(DNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),nn.LeakyReLU(0.01),nn.BatchNorm1d(512),nn.Dropout(Drop),
            nn.Linear(512, 512),nn.LeakyReLU(0.01),nn.BatchNorm1d(512),nn.Dropout(Drop),
            nn.Linear(512, 512),nn.LeakyReLU(0.01),nn.BatchNorm1d(512),nn.Dropout(Drop),
            nn.Linear(512, 512),nn.LeakyReLU(0.01),nn.BatchNorm1d(512),nn.Dropout(Drop),
            nn.Linear(512, 256),nn.LeakyReLU(0.01),nn.BatchNorm1d(256),nn.Dropout(Drop),
            nn.Linear(256, 128),nn.LeakyReLU(0.01),nn.BatchNorm1d(128),nn.Dropout(Drop),
            nn.Linear(128, 64),nn.LeakyReLU(0.01),nn.BatchNorm1d(64),nn.Dropout(Drop),
            nn.Linear(64, 32),nn.LeakyReLU(0.01),nn.BatchNorm1d(32),nn.Dropout(Drop),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)


#  Combined model
class CombinedModel(nn.Module):
    def __init__(self, eq_input_dim, dnn_input_dim ,struct,Drop=0.1):
        super(CombinedModel, self).__init__()
        self.attention = MultiHeadAttentionFeatureExtractor(eq_input_dim)
        self.attentionstruct = MultiHeadAttentionFeatureExtractor1(struct)
        self.dnn = DNN(dnn_input_dim + 40,Drop=Drop)  # 40 is the output dim of attention
    def forward(self,X_struct, x_eq, x_dnn):
        attention_features = self.attention(x_eq)
        attention_featuresstruct = self.attentionstruct(X_struct)
        combined_features = torch.cat([attention_features,attention_featuresstruct, x_dnn], dim=1)
        return self.dnn(combined_features)
    

#  Main function
def main():
    testfile = './data/building_response_testset.csv'  # Test set file
    kd_based_tl_model = './models/kd_based_tl_model/'  # KKD-based TL model path
    direct_training_baseline = './models/direct_training_baseline/'  #  Direct training baseline model path
    direct_transferred_model = './models/direct_transferred_model/'  # Direct transferred model path
    pretrained_model = './models/pretrained_model.pth'  # Pretrained model path
    model_files = [f'k{i}.pth' for i in range(1, 6)]  # 5-fold model files
    df = pd.read_csv(testfile, delimiter=',', encoding='utf-8')
    df = df.iloc[:,2:]
    df = wish(df)
    #  Feature processing
    X_cnn = np.log(df.iloc[:, 9:28].values)
    X_dnn = np.log(df.iloc[:, 3:6].values)
    X_struct = df.iloc[:, 28:].values
    Y = np.log(df.iloc[:, 0].values)
    X_cnn = torch.FloatTensor(X_cnn)
    X_dnn = torch.FloatTensor(X_dnn)
    X_struct = torch.FloatTensor(X_struct)
    Y = torch.FloatTensor(Y).unsqueeze(1)
    student_input = torch.cat([X_struct, X_cnn, X_dnn], dim=1)
    input_dim = student_input.shape[1]
    # Evaluate KD-based TL model
    mse_list = []
    r_list = []
    r2_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, test_idx) in enumerate(kfold.split(student_input)):
        X_fold = student_input[test_idx]
        y_fold = Y[test_idx]
        model = StudentModel(input_dim)
        model.load_state_dict(torch.load(kd_based_tl_model + model_files[i], map_location='cpu',weights_only=True))
        model.eval()
        with torch.no_grad():
            pred = model(X_fold).cpu().numpy().squeeze()
            true = y_fold.cpu().numpy().squeeze()
        mse = mean_squared_error(true, pred)
        r = np.corrcoef(true, pred)[0,1]
        r2 = r2_score(true, pred)
        mse_list.append(mse)
        r_list.append(r)
        r2_list.append(r2)
    model_name1 = 'KD-Based TL Model'
    df_KD_Based_TL_Model = pd.DataFrame(
    {
        "MSE_Mean": [np.mean(mse_list)],
        "MSE_Var":  [np.var(mse_list)],
        "r_Mean":   [np.mean(r_list)],
        "r_Var":    [np.var(r_list)],
        "R2_Mean":  [np.mean(r2_list)],
        "R2_Var":   [np.var(r2_list)],
    },
    index=[model_name1])
    #  Evaluate direct training baseline model
    mse_list = []
    r_list = []
    r2_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, test_idx) in enumerate(kfold.split(student_input)):
        X_fold = student_input[test_idx]
        y_fold = Y[test_idx]
        model = StudentModel(input_dim)
        model.load_state_dict(torch.load(direct_training_baseline + model_files[i], map_location='cpu',weights_only=True))
        model.eval()
        with torch.no_grad():
            pred = model(X_fold).cpu().numpy().squeeze()
            true = y_fold.cpu().numpy().squeeze()
        mse = mean_squared_error(true, pred)
        r = np.corrcoef(true, pred)[0,1]
        r2 = r2_score(true, pred)
        mse_list.append(mse)
        r_list.append(r)
        r2_list.append(r2)
    model_name2 = 'Direct-training baseline model'
    df_Direct_training_baseline_model = pd.DataFrame(
    {
        "MSE_Mean": [np.mean(mse_list)],
        "MSE_Var":  [np.var(mse_list)],
        "r_Mean":   [np.mean(r_list)],
        "r_Var":    [np.var(r_list)],
        "R2_Mean":  [np.mean(r2_list)],
        "R2_Var":   [np.var(r2_list)],
    },
    index=[model_name2])
    # Evaluate direct transferred model
    mse_list = []
    r_list = []
    r2_list = []
    d_struct = X_struct.shape[1]
    d_cnn    = X_cnn.shape[1]
    d_dnn    = X_dnn.shape[1]
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, test_idx) in enumerate(kfold.split(student_input)):
        X_fold = student_input[test_idx]
        y_fold = Y[test_idx]
        X_struct_rec, X_cnn_rec, X_dnn_rec = torch.split(
        X_fold,
        [d_struct, d_cnn, d_dnn],
        dim=1)
        pmodel = CombinedModel(eq_input_dim=19, dnn_input_dim=3 , struct=16)
        pmodel.load_state_dict(torch.load(direct_transferred_model+ model_files[i], map_location='cpu',weights_only=True))
        pmodel.eval()
        with torch.no_grad():
            pred = pmodel(X_struct_rec, X_cnn_rec, X_dnn_rec).cpu().numpy().squeeze()
            true = y_fold.cpu().numpy().squeeze()
        mse = mean_squared_error(true, pred)
        r = np.corrcoef(true, pred)[0,1]
        r2 = r2_score(true, pred)
        mse_list.append(mse)
        r_list.append(r)
        r2_list.append(r2)  
    model_name3 = 'Direct transferred pretrain model'
    df_direct_transferred_model = pd.DataFrame(
    {
        "MSE_Mean": [np.mean(mse_list)],
        "MSE_Var":  [np.var(mse_list)],
        "r_Mean":   [np.mean(r_list)],
        "r_Var":    [np.var(r_list)],
        "R2_Mean":  [np.mean(r2_list)],
        "R2_Var":   [np.var(r2_list)],
    },
    index=[model_name3])
    #  Evaluate pretrained model
    mse_list = []
    r_list = []
    r2_list = []
    d_struct = X_struct.shape[1]
    d_cnn    = X_cnn.shape[1]
    d_dnn    = X_dnn.shape[1]
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, test_idx) in enumerate(kfold.split(student_input)):
        X_fold = student_input[test_idx]
        y_fold = Y[test_idx]
        X_struct_rec, X_cnn_rec, X_dnn_rec = torch.split(
        X_fold,
        [d_struct, d_cnn, d_dnn],
        dim=1)
        pmodel = CombinedModel(eq_input_dim=19, dnn_input_dim=3 , struct=16)
        pmodel.load_state_dict(torch.load(pretrained_model, map_location='cpu',weights_only=True))
        pmodel.eval()
        with torch.no_grad():
            pred = pmodel(X_struct_rec, X_cnn_rec, X_dnn_rec).cpu().numpy().squeeze()
            true = y_fold.cpu().numpy().squeeze()
        mse = mean_squared_error(true, pred)
        r = np.corrcoef(true, pred)[0,1]
        r2 = r2_score(true, pred)
        mse_list.append(mse)
        r_list.append(r)
        r2_list.append(r2)  
    model_name4 = 'Pretrained_model'
    df_pretrained_model = pd.DataFrame(
    {
        "MSE_Mean": [np.mean(mse_list)],
        "MSE_Var":  [np.var(mse_list)],
        "r_Mean":   [np.mean(r_list)],
        "r_Var":    [np.var(r_list)],
        "R2_Mean":  [np.mean(r2_list)],
        "R2_Var":   [np.var(r2_list)],
    },
    index=[model_name4])
    # Concatenate all model results
    result_df = pd.concat([df_pretrained_model, df_direct_transferred_model, df_Direct_training_baseline_model,df_KD_Based_TL_Model])
    model_names = [
    "Pretrained model",
    "Directly transferred pretrained model",
    "Direct-training baseline model",
    "KD-based TL model"
    ]

    result_df.insert(0, " ", model_names)
    path = "data\metrics_summary.csv"
    if os.path.exists(path):
        result_df.to_csv("data\metrics_summary.csv",index=False,encoding="utf-8-sig"  )
    print("Evaluation completed. Results saved to 'data/metrics_summary.csv'.")
    return result_df

if __name__ == '__main__':
    main()
