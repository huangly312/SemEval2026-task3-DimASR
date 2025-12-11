"""
Data processing utilities for DimABSA tasks
"""

import os
import re
import json
from typing import Dict, List, Optional, Any
from datasets import Dataset, load_dataset


class PromptTemplate:
    """
    Manages prompt templates for different tasks and domains
    """
    
    # Domain-specific entity and attribute labels for Task 3
    ENTITY_ATTRIBUTE_MAP = {
        'restaurant': {
            'entity': 'RESTAURANT, FOOD, DRINKS, AMBIENCE, SERVICE, LOCATION',
            'attribute': 'GENERAL, PRICES, QUALITY, STYLE_OPTIONS, MISCELLANEOUS'
        },
        'laptop': {
            'entity': 'LAPTOP, DISPLAY, KEYBOARD, MOUSE, MOTHERBOARD, CPU, FANS_COOLING, PORTS, MEMORY, POWER_SUPPLY, OPTICAL_DRIVES, BATTERY, GRAPHICS, HARD_DISK, MULTIMEDIA_DEVICES, HARDWARE, SOFTWARE, OS, WARRANTY, SHIPPING, SUPPORT, COMPANY',
            'attribute': 'GENERAL, PRICE, QUALITY, DESIGN_FEATURES, OPERATION_PERFORMANCE, USABILITY, PORTABILITY, CONNECTIVITY, MISCELLANEOUS'
        },
        'hotel': {
            'entity': 'HOTEL, ROOMS, FACILITIES, ROOM_AMENITIES, SERVICE, LOCATION, FOOD_DRINKS',
            'attribute': 'GENERAL, PRICE, COMFORT, CLEANLINESS, QUALITY, DESIGN_FEATURES, STYLE_OPTIONS, MISCELLANEOUS'
        },
        'finance': {
            'entity': 'MARKET, COMPANY, BUSINESS, PRODUCT',
            'attribute': 'GENERAL, SALES, PROFIT, AMOUNT, PRICE, COST'
        }
    }
    
    @staticmethod
    def load_task2_prompt(prompt_file: str = "prompt/task2_prompt.txt") -> str:
        """
        Load Task 2 prompt template from file
        
        Args:
            prompt_file: Path to prompt template file
            
        Returns:
            Prompt template string
        """
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        
        # Default Task 2 prompt if file doesn't exist
        return """Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

### Instruction:
Given a textual instance [Text], extract all (A, O, VA) triplets, where:
- A is an Aspect term (a phrase describing an entity mentioned in [Text])
- O is an Opinion term
- VA is a Valence–Arousal score in the format (valence#arousal)

Valence ranges from 1 (negative) to 9 (positive),
Arousal ranges from 1 (calm) to 9 (excited).

### Example:
Input:
[Text] average to good thai food, but terrible delivery.
[Aspect] thai food, delivery

Output:
[Triplet] (thai food, average to good, 6.75#6.38), (delivery, terrible, 2.88#6.62)

### Question:
Now complete the following example:
Input:
"""
    
    @staticmethod
    def load_task3_prompt(domain: str, prompt_file: str = "prompt/task3_prompt_template.txt") -> str:
        """
        Load Task 3 prompt template from file and fill in domain-specific labels
        
        Args:
            domain: Domain name (restaurant, laptop, hotel, finance)
            prompt_file: Path to prompt template file
            
        Returns:
            Prompt template string with labels filled in
        """
        if domain not in PromptTemplate.ENTITY_ATTRIBUTE_MAP:
            raise ValueError(f"Unsupported domain: {domain}")
        
        entity_label = PromptTemplate.ENTITY_ATTRIBUTE_MAP[domain]['entity']
        attribute_label = PromptTemplate.ENTITY_ATTRIBUTE_MAP[domain]['attribute']
        
        # Try to load from file
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                template = f.read()
            # Replace placeholders
            return template.format(entity_label=entity_label, attribute_label=attribute_label)
        
        # Fallback to hardcoded template if file doesn't exist
        return f"""Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

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
"""
    
    @staticmethod
    def get_task3_prompt(domain: str) -> str:
        """
        Generate Task 3 prompt template for specific domain (deprecated, use load_task3_prompt)
        
        Args:
            domain: Domain name (restaurant, laptop, hotel, finance)
            
        Returns:
            Prompt template string
        """
        return PromptTemplate.load_task3_prompt(domain)


class DataProcessor:
    """
    Processes datasets for DimABSA Task 2 and Task 3
    """
    
    def __init__(
        self,
        task_type: str,
        domain: str,
        exclude_null: bool = True,
        prompt_file: Optional[str] = None
    ):
        """
        Args:
            task_type: "task2" or "task3"
            domain: Domain name for task3 (restaurant, laptop, hotel, finance)
            exclude_null: Whether to exclude NULL aspects
            prompt_file: Custom prompt template file path
        """
        self.task_type = task_type
        self.domain = domain
        self.exclude_null = exclude_null
        
        if task_type == "task2":
            self.instruction = PromptTemplate.load_task2_prompt(
                prompt_file if prompt_file else "prompt/task2_prompt.txt"
            )
        elif task_type == "task3":
            self.instruction = PromptTemplate.load_task3_prompt(
                domain,
                prompt_file if prompt_file else "prompt/task3_prompt_template.txt"
            )
        else:
            raise ValueError(f"Invalid task_type: {task_type}. Must be 'task2' or 'task3'")
    
    def load_train_dataset(self, data_path: str) -> Dataset:
        """
        Load and process training dataset
        
        Args:
            data_path: Path to training data file (JSONL format)
            
        Returns:
            Processed Dataset object
        """
        dataset = load_dataset("json", data_files=data_path, split="train")
        processed = dataset.map(
            self._convert_to_chat_format,
            remove_columns=dataset.column_names
        )
        
        # Filter out None values (empty samples)
        processed = processed.filter(lambda x: x['text'] is not None)
        
        return processed
    
    def load_dev_dataset(self, data_path: str) -> Dataset:
        """
        Load development dataset for inference
        
        Args:
            data_path: Path to development data file (JSONL format)
            
        Returns:
            Dataset object
        """
        return load_dataset("json", data_files=data_path, split="train")
    
    def _convert_to_chat_format(self, example: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """
        Convert raw data example to chat format for training
        
        Args:
            example: Raw data example with 'Text' and 'Quadruplet'/'Triplet'
            
        Returns:
            Dictionary with 'text' field in chat format
        """
        text = example["Text"]
        
        if self.task_type == "task2":
            quads = example.get("Triplet", example.get("Quadruplet", []))
        else:
            quads = example.get("Quadruplet", [])
        
        # Filter out NULL aspects if required
        if self.exclude_null:
            quads = [q for q in quads if q.get('Aspect') != "NULL"]
        
        # Skip empty samples
        if len(quads) == 0:
            return {"text": None}
        
        # Generate answer string
        if self.task_type == "task2":
            answer = ", ".join([
                f"({q['Aspect']}, {q['Opinion']}, {q['VA']})"
                for q in quads
            ])
        else:  # task3
            answer = ", ".join([
                f"({q['Aspect']}, {q['Category']}, {q['Opinion']}, {q['VA']})"
                for q in quads
            ])
        
        # Create prompt
        prompt = self.instruction + "[Text] " + text + "\n\nOutput:"
        
        # Format as chat template
        chat_text = f"<|user|>\n{prompt}\n<|assistant|>\n{answer}"
        
        return {"text": chat_text}
    
    def format_for_inference(self, example: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Format example for inference (chat messages)
        
        Args:
            example: Data example with 'Text' field
            
        Returns:
            List of message dictionaries for chat template
        """
        text = example["Text"]
        prompt = self.instruction + '[Text] ' + text + '\n\nOutput:'
        
        return [
            {"role": "user", "content": prompt}
        ]
    
    @staticmethod
    def extract_predictions(text: str, task_type: str) -> List[Dict[str, str]]:
        """
        Extract structured predictions from model output
        
        Args:
            text: Raw model output text
            task_type: "task2" or "task3"
            
        Returns:
            List of extracted triplets/quadruplets
        """
        result = []
        
        if task_type == "task2":
            # Pattern for (Aspect, Opinion, VA)
            pattern = r'\(([^,]+),\s*([^,]+),\s*([\d.]+#[\d.]+)\)'
            matches = re.findall(pattern, text)
            
            for aspect, opinion, va in matches:
                result.append({
                    "Aspect": aspect.strip(),
                    "Opinion": opinion.strip(),
                    "VA": va
                })
        
        elif task_type == "task3":
            # Pattern for (Aspect, Category, Opinion, VA)
            pattern = r'\(([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)'
            matches = re.findall(pattern, text)
            
            for aspect, category, opinion, va in matches:
                result.append({
                    "Aspect": aspect.strip(),
                    "Category": category.strip(),
                    "Opinion": opinion.strip(),
                    "VA": va
                })
        else:
            raise ValueError(f"Invalid task_type: {task_type}")
        
        return result
    
    @staticmethod
    def save_predictions(
        predictions: List[Dict[str, Any]],
        output_path: str
    ):
        """
        Save predictions to JSONL file
        
        Args:
            predictions: List of prediction dictionaries
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in predictions:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Predictions saved to: {output_path}")


