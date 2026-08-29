import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_absolute_error
from sklearn.neural_network import MLPClassifier, MLPRegressor

from src.data_generator import generate_student_dataset
from src.preprocessing import DataPreprocessor
from src.feature_engineering import add_engineered_features
from src.train_ml import FEATURE_COLUMNS

# Safely check if PyTorch is functional on this OS environment
HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    # Test tensor instantiation to catch C10 DLL issues
    _test_t = torch.tensor([1.0])
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def train_dl_model(data_path: str = None, models_dir: str = "models/saved_models", epochs: int = 80) -> Dict[str, Any]:
    os.makedirs(models_dir, exist_ok=True)
    
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = generate_student_dataset(n_samples=1500)
        
    df_feat = add_engineered_features(df)
    
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    if os.path.exists(preprocessor_path):
        preprocessor = DataPreprocessor.load(preprocessor_path)
        df_proc = preprocessor.transform(df_feat)
    else:
        preprocessor = DataPreprocessor()
        df_proc = preprocessor.fit_transform(df_feat)
        preprocessor.save(preprocessor_path)
        
    X = df_proc[FEATURE_COLUMNS].values.astype(np.float32)
    y_reg = df_proc["Final_Score"].values.astype(np.float32)
    y_cls = (df_proc["Target"] == "Pass").values.astype(np.int32)
    
    X_train, X_test, y_train_cls, y_test_cls = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    
    if HAS_TORCH:
        try:
            class StudentPerformanceNN(nn.Module):
                def __init__(self, input_dim: int):
                    super(StudentPerformanceNN, self).__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 32),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(32, 1)
                    )
                def forward(self, x):
                    return self.net(x)

            train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train_cls, dtype=torch.float32).unsqueeze(1))
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            input_dim = X.shape[1]
            model = StudentPerformanceNN(input_dim=input_dim)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
            
            model.train()
            for epoch in range(epochs):
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
            model.eval()
            with torch.no_grad():
                test_x = torch.tensor(X_test)
                logits = model(test_x)
                probs = torch.sigmoid(logits).numpy().flatten()
                preds = (probs >= 0.5).astype(int)
                
            acc = accuracy_score(y_test_cls, preds)
            f1 = f1_score(y_test_cls, preds, zero_division=0)
            auc = roc_auc_score(y_test_cls, probs)
            
            model_save_path = os.path.join(models_dir, "pytorch_student_model.pt")
            torch.save({"input_dim": input_dim, "state_dict": model.state_dict()}, model_save_path)
            
            print(f"Deep Learning Neural Network (PyTorch) Complete! Acc: {acc:.4f}, F1: {f1:.4f}")
            return {"Framework": "PyTorch", "Accuracy": round(acc, 4), "F1 Score": round(f1, 4), "ROC-AUC": round(auc, 4)}
        except Exception as e:
            print(f"PyTorch execution error: {e}. Falling back to Scikit-Learn MLP Neural Network.")
            
    # Scikit-Learn MLP Neural Network
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42, early_stopping=True)
    mlp.fit(X_train, y_train_cls)
    preds = mlp.predict(X_test)
    probs = mlp.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test_cls, preds)
    f1 = f1_score(y_test_cls, preds, zero_division=0)
    auc = roc_auc_score(y_test_cls, probs)
    
    joblib.dump(mlp, os.path.join(models_dir, "mlp_neural_network.joblib"))
    print(f"Deep Learning Multi-Layer Perceptron (Neural Network) Complete! Acc: {acc:.4f}, F1: {f1:.4f}")
    
    return {
        "Framework": "Scikit-Learn MLP Neural Network",
        "Accuracy": round(acc, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(auc, 4)
    }

if __name__ == "__main__":
    results = train_dl_model()
    print("DL Results:", results)
