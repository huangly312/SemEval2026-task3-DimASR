直接在主目录下运行`python main.py`即可.

可修改参数:
常见`main.py` L12-L18 行

- `subtask = "subtask_2"# subtask_2 or subtask_3`
- `task = "task2" # task2 or task3`
- `lang = "zho" #chang the language you want to test`
- `domain = "restaurant" #change what domain you want to test`
- `model_id = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit" # you can change the model here`
- `exclude_NULL = True` # 是否排除 NULL 值,True 为排除,False 为不排除

注意: unsloth 的环境较为严苛,可以查看:https://github.com/unslothai/unsloth
我的配置为:
PyTorch 2.8.0

Python 3.12(ubuntu22.04)

CUDA 12.8

Transformers 4.57.2

unsloth 2025.11.4

不建议使用其他任何与上述版本号不同的配置,否则可能报错.
