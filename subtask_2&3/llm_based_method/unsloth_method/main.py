import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
import json
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


#task config
subtask = "subtask_2"# subtask_2 or subtask_3
task = "task2" # task2 or task3
lang = "zho" #chang the language you want to test
domain = "restaurant" #change what domain you want to test
model_id = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit" # you can change the model here

exclude_NULL = True

train_path = f"data/raw_track_a/{subtask}/{lang}/{lang}_{domain}_train_alltasks.jsonl"
predict_path = f"data/raw_track_a/{subtask}/{lang}/{lang}_{domain}_dev_{task}.jsonl"

#load train data from local path
dataset = load_dataset("json", data_files=train_path)

# task 2 prompt template covert
if task == "task2":
  with open("task2.txt", "r") as f:
    instruction = f.read()

  def convert(x):
      text = x["Text"]
      quads = x.get("Quadruplet", [])
      global exclude_NULL
      if exclude_NULL:
        quads = [q for q in quads if q['Aspect'] != "NULL"]
      if len(quads) == 0: return None
      answer = ", ".join([
          f"({q['Aspect']}, {q['Opinion']}, {q['VA']})"
          for q in quads
      ])
      prompt = instruction + "[Text] " + text + "\n\nOutput:"
      return {"text": f"<|user|>\n{prompt}\n<|assistant|>\n{answer}"}
  
# task 3 prompt template covert, with task3 predefine entity and attribute labels.
elif task == "task3":
  rest_entity = 'RESTAURANT, FOOD, DRINKS, AMBIENCE, SERVICE, LOCATION'
  rest_attribute = 'GENERAL, PRICES, QUALITY, STYLE_OPTIONS, MISCELLANEOUS'

  laptop_entity = 'LAPTOP, DISPLAY, KEYBOARD, MOUSE, MOTHERBOARD, CPU, FANS_COOLING, PORTS, MEMORY, POWER_SUPPLY, OPTICAL_DRIVES, BATTERY, GRAPHICS, HARD_DISK, MULTIMEDIA_DEVICES, HARDWARE, SOFTWARE, OS, WARRANTY, SHIPPING, SUPPORT, COMPANY'
  laptop_attribute = 'GENERAL, PRICE, QUALITY, DESIGN_FEATURES, OPERATION_PERFORMANCE, USABILITY, PORTABILITY, CONNECTIVITY, MISCELLANEOUS'

  hotel_entity = 'HOTEL, ROOMS, FACILITIES, ROOM_AMENITIES, SERVICE, LOCATION, FOOD_DRINKS'
  hotel_attribute = 'GENERAL, PRICE, COMFORT, CLEANLINESS, QUALITY, DESIGN_FEATURES, STYLE_OPTIONS, MISCELLANEOUS'

  finance_entity = 'MARKET, COMPANY, BUSINESS, PRODUCT'
  finance_attribute = 'GENERAL, SALES, PROFIT, AMOUNT, PRICE, COST'

  entity_attribute_map = {
      'restaurant': (rest_entity, rest_attribute),
      'laptop': (laptop_entity, laptop_attribute),
      'hotel': (hotel_entity, hotel_attribute),
      'finance': (finance_entity, finance_attribute),
  }

  entity_label, attribute_label = entity_attribute_map[domain]

  instruction = f'''Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

### Instruction:
Given a textual instance [Text], extract all (A, C, O, VA) quadruplets, where:
- A is an Aspect term (a phrase describing an entity mentioned in [Text])
- C is a Category label (e.g. FOOD#QUALITY)
- O is an Opinion term
- VA is a Valence–Arousal score in the format (valence#arousal)

Valence ranges from 1 (negative) to 9 (positive),
Arousal ranges from 1 (calm) to 9 (excited).

### Label constraints:
[Entity Labels] ({entity_label})
[Attribute Labels] ({attribute_label})

### Example:
Input:
[Text] average to good thai food, but terrible delivery.

Output:
[Quadruplet] (thai food, FOOD#QUALITY, average to good, 6.75#6.38),
             (delivery, SERVICE#GENERAL, terrible, 2.88#6.62)

### Question:
Now complete the following example:
Input:
'''

  def convert(x):
      text = x["Text"]
      quads = x.get("Quadruplet", [])
      global exclude_NULL
      if exclude_NULL:
        quads = [q for q in quads if q['Aspect'] != "NULL"]
      if len(quads) == 0: return None
      answer = ", ".join([
          f"({q['Aspect']}, {q['Category']}, {q['Opinion']}, {q['VA']})"
          for q in quads
      ])
      prompt = instruction + "[Text] " + text + "\n\nOutput:"
      return {"text": f"<|user|>\n{prompt}\n<|assistant|>\n{answer}"}


# covert dataset to train template
train_dataset = dataset["train"].map(convert)
print(len(train_dataset))

# fine-tuning

# tokenizer and model setting
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = 512, # DimASBA Task usually less then 512 tokens.
    load_in_4bit = True,
)

# lora setting
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0.05,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
)



trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = 512,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 20,
        num_train_epochs = 2, # epoches
        learning_rate = 1e-4, #　learning rate
        logging_steps = 50,
        save_steps = 100,
        fp16 = False,
        bf16 = True,
        report_to = "none",
        output_dir = f"./Lora/{subtask}_{lang}_{domain}_{'no_NULL' if exclude_NULL else 'has_NULL'}", # save Lora
    ),
)

trainer.train()


# load dev json to predict
predict_dataset = load_dataset("json", data_files=predict_path)

# convert text to prompt
def format_dataset(x):
    text = x["Text"]
    final_prompt = instruction + '[Text] ' + text + '\n\nOutput:'
    return [
        {"role": "user", "content": final_prompt}
    ]

# extract answer
def extract_answer(text,task):
  result = []
  if task == "task2":
    pattern = r'\(([^,]+),\s*([^,]+),\s*([\d.]+#[\d.]+)\)'
    matches = re.findall(pattern, text)

    for aspect, opinion, va in matches:
        meta_triplet = {}
        meta_triplet["Aspect"] = aspect.strip()
        meta_triplet["Opinion"] = opinion.strip()
        meta_triplet["VA"] = va
        result.append(meta_triplet)

  elif task == "task3":
    pattern = r'\(([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)'
    matches = re.findall(pattern, text)

    for aspect, category, opinion, va in matches:
        meta_quadra = {}
        meta_quadra["Aspect"] = aspect.strip()
        meta_quadra["Category"] = category.strip()
        meta_quadra["Opinion"] = opinion.strip()
        meta_quadra["VA"] = va
        result.append(meta_quadra)
  else:
    raise ValueError("Invalid task")

  return result

# Perform inference
results = []
for i, sample in enumerate(predict_dataset["train"]):
    messages = format_dataset(sample)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    result = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        max_new_tokens=1024,
        temperature=0.7, top_p=0.8, top_k=20,
    )

    decoded = tokenizer.decode(result[0])
    extracted_text = decoded.split("\n")[-1]

    key = "Triplet" if task == "task2" else "Quadruplet"

    dump_data = {
        "ID": sample.get("ID", f"sample_{i}"),
        "Text": sample["Text"],
        key: extract_answer(extracted_text, task),
    }

    print(dump_data)
    results.append(dump_data)

# resolve output name
out_name = f"pred_{lang}_{domain}_{'no_NULL' if exclude_NULL else 'has_NULL'}.jsonl"

# ensure output folder
os.makedirs(subtask, exist_ok=True)

# JSONL file path
jsonl_path = os.path.join(subtask, out_name)

# write JSONL
with open(jsonl_path, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")