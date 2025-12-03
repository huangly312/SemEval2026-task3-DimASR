写在前面:
- 因为结构的原因需要将数据集迁移到pipeline_based_method/data文件夹下才方便运行
- 注意,如果选择subtask_3任务,模型会同步预测出一个subtask_2的结果,为了区分我已经对结果命名进行了标注,训练的时候可以比较
- 为了更方便地运行, 直接启动run.sh脚本即可,相关的参数可以进行调试.
- 为了确定程序不会报错,需要在`run_task2&3_trainer_multilingual.py`中查找`out_put_file_name_map`的定义,保证数据集的输出名称包含在内,否则会报`key error`错误,直接仿照结构添加即可.
- 如果需要比较排除NULL前后的表现,修改脚本中`exclude_NULL`参数为`True`或`False`即可.
- 运行必须要GPU Cuda环境,如果试运行需要在脚本中添加参数`--gpu False`禁用GPU.
- 以上脚本均指代`run.sh`脚本

Starter Kit (Subtask 2 & 3)

Train and run the Dimensional Aspect-Based Sentiment Analysis (DimABSA) model for multilingual datasets.  
Supports Task 2 (Triplet Extraction) and Task 3 (Quadruplet Extraction).

#---- Folder Structure ----#
./data/              # Training and inference data (.jsonl)
./model/             # Saved model checkpoints
./log/               # Training logs
./tasks/subtask_2/   # Task 2 output files
./tasks/subtask_3/   # Task 3 output files


#---- Note ----#
Before running:
“Place all dataset files in ./data/ and ensure required dependencies (transformers, torch, etc.) are installed.”

After running:
“Predictions will be saved automatically to ./tasks/subtask_2/ and ./tasks/subtask_3/ depending on the task.”


#----Key Arguments----#
--task <int>
Task type: 2 or 3
2 → outputs triplets to ./tasks/subtask_2/
3 → outputs triplets and quadruplets to both ./tasks/subtask_2/ and ./tasks/subtask_3/

--domain <str>
Dataset domain (res | lap | hot | fin)

--language <str>
Dataset language (eng | zho)

--train_data <str>
Training data filename under ./data/

--infer_data <str>
Inference (test) data filename under ./data/

--bert_model_type <str>
Pretrained BERT model name or local path
Example: bert-base-multilingual-uncased

--mode <str>
Operation mode:
train → trains model and performs inference
inference → loads trained model and performs prediction only

--epoch_num <int>
Number of training epochs (default: 3)

--batch_size <int>
Training batch size (default: 4)

--learning_rate <float>
Learning rate for non-BERT parameters (default: 1e-3)

--tuning_bert_rate <float>
Learning rate for BERT fine-tuning (default: 1e-5)

--inference_beta <float>
Confidence threshold for prediction filtering (default: 0.9)

--gpu <bool>
Enable CUDA (default: True)

--reload <bool>
Resume training from checkpoint (default: False)


#---- Task 2 – Triplet Extraction ----#
{"ID": "res_dev_1", "Triplet": [
  {"Aspect": "food", "Opinion": "great", "VA": "7.8#7.2"}
]}

#---- Task 3 – Quadruplet Extraction ----#
{"ID": "res_dev_1", "Quadruplet": [
  {"Aspect": "food", "Category": "FOOD#QUALITY", "Opinion": "great", "VA": "7.8#7.2"}
]}


#---- Model Training Example----#
python run_task2&3_trainer_multilingual.py \
  --task 3 \
  --domain res \
  --language eng \
  --train_data eng_restaurant_train_alltasks.jsonl \
  --infer_data eng_restaurant_dev_task2.jsonl \
  --bert_model_type bert-base-multilingual-uncased \
  --mode train

#---- Model Inference Example----#
python run_task2&3_trainer_multilingual.py \
  --task 3 \
  --domain res \
  --language eng \
  --train_data eng_restaurant_train_alltasks.jsonl \
  --infer_data eng_restaurant_dev_task2.jsonl \
  --bert_model_type bert-base-multilingual-uncased \
  --mode inference