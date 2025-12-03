import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from data_processing import load_jsonl, jsonl_to_df, df_to_jsonl
from evaluation import get_prd, evaluate_predictions_task1
from model import TransformerVARegressor
from data_processing import VADataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import yaml
import json

## from yaml load config
with open("subtask_1/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print(f"Configuration loaded successfully: {config}\n")
subtask = config["subtask"]
task = config["task"]

lang = config["lang"]
domain = config["domain"]
model_name = config["model_name"]
# 如果配置了 model_path，优先使用本地路径加载模型
model_path = config.get("model_path", model_name)  # 如果没配置 model_path，就用 model_name
use_local_only = config.get("use_local_only", False)
lr = float(config["lr"])
epochs = int(config["epochs"])

if (lang=="zho" and domain=="finance") or (lang=="jpn" and domain=="finance"):
    train_path=f"data/raw_track_a/{subtask}/{lang}/{lang}_{domain}_train_{task}.jsonl"
else:
    train_path=f"data/raw_track_a/{subtask}/{lang}/{lang}_{domain}_train_alltasks.jsonl"
predict_path = f"data/raw_track_a/{subtask}/{lang}/{lang}_{domain}_dev_{task}.jsonl"
save_path = f"{subtask}/output/pred_{lang}_{domain}({model_name}-{lr:.1e}-{epochs}).jsonl"
best_model_path = f"{subtask}/checkpoints/best_model({lang}-{domain}-{model_name}-{lr:.1e}-{epochs}).pth"
training_log_path = f"{subtask}/logs/training_log({lang}-{domain}-{model_name}-{lr:.1e}-{epochs}).json"

if __name__ == "__main__":
    
    ## create necessary directories
    os.makedirs(f"{subtask}/checkpoints", exist_ok=True)
    os.makedirs(f"{subtask}/output", exist_ok=True)
    os.makedirs(f"{subtask}/logs", exist_ok=True)

    ## load data
    train_raw = load_jsonl(train_path)
    predict_raw = load_jsonl(predict_path)
    train_df = jsonl_to_df(train_raw)
    predict_df = jsonl_to_df(predict_raw)

    ## split data
    train_df, dev_df = train_test_split(train_df, test_size=0.1, random_state=42)

    ## tokenize data
    print(f"Loading tokenizer: {model_path} (local_only={use_local_only})")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=use_local_only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    train_dataset = VADataset(train_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    dev_dataset = VADataset(dev_df, tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=True)

    ## load model and optimizer
    print(f"Loading model: {model_path} (local_only={use_local_only})")
    model = TransformerVARegressor(model_path=model_path, use_local_only=use_local_only).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    print(f"Model loaded successfully\n")

    ## training loop with best model selection
    best_score = float('inf')
    training_history = []
    
    print(f"\n{'='*80}")
    print(f"Training {model_name} on {lang}-{domain} dataset")
    print(f"{'='*80}\n")
    
    for epoch in range(epochs):
        train_loss = model.train_epoch(train_loader, optimizer, loss_fn, device)
        val_loss = model.eval_epoch(dev_loader, loss_fn, device)
        
        pred_v, pred_a, gold_v, gold_a = get_prd(model, dev_loader, device, type="dev")
        eval_metrics = evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v)
        
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": float(f"{train_loss:.6f}"),
            "val_loss": float(f"{val_loss:.6f}"),
            "PCC_V": float(f"{eval_metrics['PCC_V']:.6f}"),
            "PCC_A": float(f"{eval_metrics['PCC_A']:.6f}"),
            "RMSE_VA": float(f"{eval_metrics['RMSE_VA']:.6f}")
        }
        training_history.append(epoch_log)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"  PCC_V: {eval_metrics['PCC_V']:.6f} | PCC_A: {eval_metrics['PCC_A']:.6f} | RMSE_VA: {eval_metrics['RMSE_VA']:.6f}")
        
        current_score = eval_metrics['RMSE_VA']
        
        if current_score < best_score:
            best_score = current_score
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score,
                'eval_metrics': eval_metrics
            }, best_model_path)
            print(f"  >>> Best model saved (RMSE_VA: {best_score:.6f})")
        print()
    
    with open(training_log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'model_name': model_name,
                'lang': lang,
                'domain': domain,
                'lr': lr,
                'epochs': epochs
            },
            'best_score': float(f"{best_score:.6f}"),
            'training_history': training_history
        }, f, indent=2, ensure_ascii=False)
    
    print(f"{'='*80}")
    print(f"Training completed. Best RMSE_VA: {best_score:.6f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training log saved to: {training_log_path}")
    print(f"{'='*80}\n")
    
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}\n")


    print("Generating predictions on test set...")
    pred_dataset = VADataset(predict_df, tokenizer)
    pred_loader = DataLoader(pred_dataset, batch_size=64, shuffle=False)
    pred_v, pred_a = get_prd(model, pred_loader, device, type="pred")

    predict_df["Valence"] = pred_v
    predict_df["Arousal"] = pred_a

    df_to_jsonl(predict_df, save_path)
    print(f"Predictions saved to: {save_path}\n")

