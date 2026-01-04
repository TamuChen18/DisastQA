import json
import os
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time
import re
from dotenv import load_dotenv
import random

# Get base directory (project root) - can be customized via environment variable
_BASE_DIR = os.getenv("DISASTQA_BASE_DIR", None)
if _BASE_DIR is None:
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# model configurations - updated to actual downloaded models
# Note: Model paths are relative to the project root. Users should download models
# and update these paths, or set DISASTQA_BASE_DIR environment variable.
MODEL_CONFIGS = {
    "llama-3-8b": {
        "path": os.path.join(_BASE_DIR, "models", "llama-3-8b"),
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "max_new_tokens": 32,
        "generation_config": {
            # "temperature": 1e-5,  # very low temperature, close to deterministic
            "do_sample": False,   # greedy decoding
            "max_new_tokens": 32
        }
    },
    "llama-3.2-3b-instruct": {
        "path": os.path.join(_BASE_DIR, "models", "llama-3.2-3b-instruct"),
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "max_new_tokens": 32,
        "generation_config": {
            # "temperature": 1e-5,  # very low temperature, close to deterministic
            "do_sample": False,   # greedy decoding
            "max_new_tokens": 32
        }
    },
    "Mistral-7B-Instruct-v0.2": {
        "path": os.path.join(_BASE_DIR, "models", "Mistral-7B-Instruct-v0.2"),
        "max_sequence_length": 8192,  # 32K max, use 8K as safe value
        "torch_dtype": torch.float16,
        "max_new_tokens": 32,  # usually short answer for MCQ (32 tokens)
        "device_map": "auto",
        "generation_config": {
            "temperature": 1e-5,  # very low temperature, close to deterministic
            "do_sample": False,   # greedy decoding
            "max_new_tokens": 32
        }
    },
    "qwen-2.5-3b-instruct": {
        "path": os.path.join(_BASE_DIR, "models", "qwen-2.5-3b-instruct"),
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "max_new_tokens": 32,
        "generation_config": {
            # "temperature": 1e-5,
            "do_sample": False,
            "max_new_tokens": 32
        }
    },
    "qwen-3-4b": {
        "path": os.path.join(_BASE_DIR, "models", "qwen-3-4b"),
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "max_new_tokens": 32,
            "generation_config": {
            # "temperature": 1e-5,
            "do_sample": False,
            "max_new_tokens": 32
        }
    },
    "qwen-3-8b": {
        "path": os.path.join(_BASE_DIR, "models", "qwen-3-8b"),
        "max_new_tokens": 32,  # usually short answer for MCQ (32 tokens)
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "generation_config": {
            # "temperature": 1e-5,  # very low temperature, close to deterministic
            "do_sample": False,   # greedy decoding
            "max_new_tokens": 32
        }
    },
    "deepseek-v3-7b": {
        "path": os.path.join(_BASE_DIR, "models", "deepseek-v3-7b"),
        "max_new_tokens": 32,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "generation_config": {
            # "temperature": 1e-5,  # very low temperature, close to deterministic
            "do_sample": False,    # Enable sampling but with very low temperature
            "max_new_tokens": 32
        }
    },
    "phi-2": {
        "path": os.path.join(_BASE_DIR, "models", "phi-2"),
        "max_new_tokens": 32,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 2048,  # actually limited to 2K
        "generation_config": {
            # "temperature": 1e-5,  # very low temperature, close to deterministic
            "do_sample": False,    # greedy decoding
            "max_new_tokens": 32
        }
    },
    "gemma-7b": {
        "path": os.path.join(_BASE_DIR, "models", "gemma-7b"),
        "max_new_tokens": 32,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 2048,
        "generation_config": {
            # "temperature": 1e-5,  # very low temperature, close to deterministic
            "do_sample": False,    # greedy decoding
            "max_new_tokens": 32
        }
    },
    "Llama-3.2-1B-Instruct": {
        "path": os.path.join(_BASE_DIR, "models", "Llama-3.2-1B-Instruct"),
        "torch_dtype": torch.float16,
        "max_new_tokens": 32,
        "device_map": "auto",
        "max_sequence_length": 2048,
        "generation_config": {
            # "temperature": 1e-5,
            "do_sample": False,
            "max_new_tokens": 32
        }
    },
    "qwen-3-0.6b": {
        "path": os.path.join(_BASE_DIR, "models", "qwen-3-0.6b"),
        "torch_dtype": torch.float16,
        "max_new_tokens": 32,
        "device_map": "auto",
        "max_sequence_length": 8192,  # 40K max, use 8K as safe value
        "generation_config": {
            # "temperature": 1e-5,  # very low temperature, close to deterministic  
            "do_sample": False,
            "max_new_tokens": 32
        }
    },
    "Hunyuan-7B-Instruct": {
        "path": os.path.join(_BASE_DIR, "models", "Hunyuan-7B-Instruct"),
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
        "max_sequence_length": 8192,
        "max_new_tokens": 32,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 32
        }
    },
    "Hunyuan-4B-Instruct": {
        "path": os.path.join(_BASE_DIR, "models", "Hunyuan-4B-Instruct"),
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
        "max_sequence_length": 8192,
        "max_new_tokens": 32,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 32
        }
    },
    "Yi-6B-Chat": {
        "path": os.path.join(_BASE_DIR, "models", "Yi-6B-Chat"),
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "max_new_tokens": 32,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 32
        }
    },

    "Hunyuan-0.5B-Instruct": {
        "path": os.path.join(_BASE_DIR, "models", "Hunyuan-0.5B-Instruct"),
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
        "max_sequence_length": 1024,
        "max_new_tokens": 32,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 32
        }
    },
    "AceMath-1.5B-Instruct": {
        "path": os.path.join(_BASE_DIR, "models", "AceMath-1.5B-Instruct"),
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 2048,
        "max_new_tokens": 32,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 32
        }
    },
    "Falcon3-1B-Instruct": {
        "path": os.path.join(_BASE_DIR, "models", "Falcon3-1B-Instruct"),
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 2048,
        "max_new_tokens": 32,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 32
        }
    }
}

class LocalModelEvaluator:
    """Evaluator for local models"""
    
    def __init__(self, test_set_path: str, model_name: str, model=None, tokenizer=None):
        """Initialize evaluator"""
        self.test_set_path = test_set_path
        self.model_name = model_name
        self.model_config = MODEL_CONFIGS[model_name]
        self.setting = os.path.basename(test_set_path).split('_')[0]  # base/golden/mix
        
        # Load passages_by_score data for mix setting (to construct 5 passages)
        self.passages_data = {}
        if self.setting == "mix":
            self._load_passages_data()
        
        # print initial GPU memory status
        print("\nInitial GPU Memory Status:")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")
        print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
        print(f"Cached Memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")
        
        # Load test set
        with open(test_set_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Convert data format
        self.test_set = []
        for item in raw_data:
            if 'multiple_choice' in item and 'gpt40' in item['multiple_choice']:
                content = item['multiple_choice']['gpt40']['content']
                passage = item.get('passage', '')  # Save original passage (for golden/base)
                original_query = item.get('original_query', '')  # For mix setting to load passages_by_score
                
                # For mix setting, construct 5 passages (1 golden + 4 distractors)
                if self.setting == "mix" and original_query:
                    five_passages = self._construct_five_passages(original_query)
                else:
                    five_passages = None
                
                processed_item = {
                    'question': content['question'],
                    'options': content['options'],  # Keep original options
                    'correct_answer': content['correct_option'],
                    'context': [passage] if self.setting in ['golden'] and passage else [],  # Single passage for golden
                    'passage': passage,  # Save passage for reference
                    'five_passages': five_passages,  # 5 passages for mix setting
                    'original_query': original_query  # Save original_query for mix
                }
                self.test_set.append(processed_item)
        
        # Verify data format
        required_keys = ['question', 'options', 'correct_answer']
        for i, item in enumerate(self.test_set):
            missing_keys = [key for key in required_keys if key not in item]
            if missing_keys:
                print(f"Warning: Item {i} is missing keys: {missing_keys}")
        
        # Load model and tokenizer if not provided
        if model is None or tokenizer is None:
            print(f"Loading model {model_name} from {self.model_config['path']}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config['path'],
                trust_remote_code=True
            )
            # set padding token and padding side
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'  # set padding side to left
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config['path'],
                torch_dtype=self.model_config['torch_dtype'],
                device_map=self.model_config['device_map'],
                trust_remote_code=True
            )
            print("Model loaded successfully!")
        else:
            self.model = model
            self.tokenizer = tokenizer
            # set padding token and padding side
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'  # set padding side to left
        
        # Initialize generation config - comment out to avoid DynamicCache error
        if self.model_config.get("generation_config") is not None:
            raw_config = self.model_config["generation_config"]
            gen_config = raw_config if isinstance(raw_config, dict) else raw_config.to_dict()
            gen_config["eos_token_id"] = self.tokenizer.eos_token_id
            gen_config["pad_token_id"] = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            self.model_config["generation_config"] = GenerationConfig(**gen_config)
        
        # Set random seed for reproducibility
        # random.seed(42)
        torch.cuda.empty_cache()  # Clean up GPU memory

    def _load_passages_data(self):
        """Load passages_by_score data from DATA/DATA/data_prepare for mix setting"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(os.path.dirname(script_dir))
            data_prepare_dir = os.path.join(base_dir, "DATA", "DATA", "data_prepare")
            
            if not os.path.exists(data_prepare_dir):
                print(f"Warning: data_prepare directory not found at {data_prepare_dir}")
                return
            
            # Load all *_by_score.json files
            import glob
            json_files = glob.glob(os.path.join(data_prepare_dir, "*_by_score.json"))
            
            print(f"Loading passages data from {len(json_files)} files...")
            
            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    for item in data:
                        user_query = item.get('user_query', '')
                        if user_query:
                            passages_by_score = item.get('passages_by_score', {})
                            if passages_by_score:
                                self.passages_data[user_query] = passages_by_score
            
            print(f"Loaded passages data for {len(self.passages_data)} queries")
            
        except Exception as e:
            print(f"Error loading passages data: {e}")
            import traceback
            traceback.print_exc()
    
    def _construct_five_passages(self, original_query: str) -> List[str]:
        """Construct 5 passages for mix setting: 1 golden (score=3) + 4 distractors (one from each score 0,1,2,3)"""
        if original_query not in self.passages_data:
            print(f"Warning: No passages data found for query: {original_query[:50]}...")
            return None
        
        passages_by_score = self.passages_data[original_query]
        
        # Get golden passage (score=3)
        golden_passages = passages_by_score.get('3', [])
        if not golden_passages:
            print(f"Warning: No golden passages (score=3) found for query: {original_query[:50]}...")
            return None
        
        golden_passage = golden_passages[0]  # Use first golden passage
        
        # Get one distractor from each score (0, 1, 2)
        distractors = []
        for score in ['0', '1', '2']:
            score_passages = passages_by_score.get(score, [])
            if score_passages:
                distractors.append(random.choice(score_passages))
            else:
                # If no passage for this score, use another from available scores
                for alt_score in ['0', '1', '2']:
                    if alt_score != score and alt_score in passages_by_score:
                        alt_passages = passages_by_score[alt_score]
                        if alt_passages:
                            distractors.append(random.choice(alt_passages))
                            break
        
        # Add one more distractor if we don't have 4 yet (can be from any low score)
        if len(distractors) < 4:
            all_low_passages = []
            for score in ['0', '1', '2']:
                all_low_passages.extend(passages_by_score.get(score, []))
            if all_low_passages:
                while len(distractors) < 4 and all_low_passages:
                    distractors.append(random.choice(all_low_passages))
                    all_low_passages.remove(distractors[-1])
        
        # Combine: 1 golden + 4 distractors = 5 passages total
        five_passages = [golden_passage] + distractors[:4]
        
        # Randomly shuffle the order (so golden is not always first)
        random.shuffle(five_passages)
        
        return five_passages

    def _load_golden_lookup(self):
        """Load golden results as lookup table for mix optimization"""
        try:
            # Build golden result file path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(os.path.dirname(script_dir))
            golden_path = os.path.join(base_dir, "DATA", "local_MCQ", self.model_name, "golden_test.json")
            
            if os.path.exists(golden_path):
                print(f"Loading golden results from: {golden_path}")
                with open(golden_path, 'r', encoding='utf-8') as f:
                    golden_data = json.load(f)
                
                # Build lookup table
                for item in golden_data:
                    # Use question + passage as key
                    question = item['question']
                    passage = item.get('passage', '')  # Get passage from original data
                    key = f"{question}_{passage}"
                    self.golden_lookup[key] = {
                        'model_answer': item['model_answer'],
                        'model_explanation': item.get('model_explanation', ''),
                        'is_correct': item['is_correct'],
                        'correct_answer': item['correct_answer']
                    }
                
                print(f"Loaded {len(self.golden_lookup)} golden results for lookup")
            else:
                print(f"Warning: Golden results not found at {golden_path}")
                print("Will process mix without optimization")
                
        except Exception as e:
            print(f"Error loading golden lookup: {e}")
            print("Will process mix without optimization")
    
    def _truncate_context_for_small_model(self, context: List[str], max_tokens_per_passage: int = 200) -> List[str]:
        """truncate context to fit the token limit of small models"""
        if not self.model_config.get('is_small_model', False):
            return context
        
        truncated_context = []
        for passage in context:
            # simple truncation by character count, approximately 4 characters = 1 token
            if len(passage) > max_tokens_per_passage * 4:
                passage = passage[:max_tokens_per_passage * 4] + "..."
            truncated_context.append(passage)
        
        return truncated_context

    def generate_answer(self, query: str, context: List[str] = None, options: List[str] = None, five_passages: List[str] = None) -> str:
        """Generate answer for a question"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Format options as A, B, C, D
                options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                
                # Build prompt based on setting
                if five_passages and len(five_passages) == 5:
                    # Mix setting: 5 passages with selection instruction
                    passages_text = "\n\n".join([f"Passage {i+1}: {passage}" for i, passage in enumerate(five_passages)])
                    prompt = f"""You are given 5 passages (some may be irrelevant). You must select ONLY ONE passage that is most relevant to answer the question. Write the passage used: Passage: <single number between 1 and 5> (only one number, no commas or multiple numbers), then provide your answer.

{passages_text}

Question: {query}
Options:
{options_str}

Answer:"""
                elif context:
                    # Golden setting: single passage
                    context = self._truncate_context_for_small_model(context)
                    context_str = "\n".join(context) if isinstance(context, list) else context
                    prompt = f"""Passage: {context_str}\n\nQuestion: {query}\nOptions:\n{options_str}\n\nAnswer:"""
                else:
                    # Base setting: no passage
                    prompt = f"""Question: {query}\nOptions:\n{options_str}\n\nAnswer:"""
                
                print("\nGenerating response...")
                print(f"Prompt length: {len(prompt)}")
                
                # Generate response
                print("Tokenizing input...")
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.model_config['max_sequence_length']
                ).to(self.model.device)
                
                # remove token_type_ids to avoid compatibility issues with some models
                if 'token_type_ids' in inputs:
                    del inputs['token_type_ids']
                
                print(f"Input length: {len(inputs['input_ids'][0])}")
                print("Starting generation...")
                
                # Generate with model-specific parameters
                with torch.no_grad():
                    try:
                        # Add progress indicator
                        print("Generating tokens...", end="", flush=True)
                        start_time = time.time()
                        # use simpler generation parameters to avoid version compatibility issues
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=self.model_config["max_new_tokens"],
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                        end_time = time.time()
                        print(f"\nGeneration completed in {end_time - start_time:.2f} seconds!")
                    except Exception as e:
                        print(f"Error during generation: {str(e)}")
                        raise
                
                print(f"Output length: {len(outputs[0])}")
                
                # Decode and return response
                print("Decoding response...")
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the prompt from the response
                response = response[len(prompt):].strip()
                
                print("Generated response:")
                print(response)
                
                if not response:
                    print("Warning: Empty response generated!")
                    if attempt < max_retries - 1:
                        print(f"Retrying... (attempt {attempt + 1}/{max_retries})")
                        continue
                    else:
                        raise ValueError("Failed to generate non-empty response after all retries")
                
                return response
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"\nAttempt {attempt + 1} failed: {str(e)}")
                time.sleep(1)  # Wait for 1 second before retrying

    def parse_answer(self, response: str) -> Dict[str, str]:
        """Parse model's answer from response"""
        try:
            # first try to match "Answer: X" format
            answer_match = re.search(r'Answer:\s*([A-D])', response)
            if answer_match:
                answer = answer_match.group(1)
            else:
                # if not found standard format, try to match A/B/C/D
                answer_match = re.search(r'[A-D]', response)
                if answer_match:
                    answer = answer_match.group(0)
                else:
                    # if still not found, try to match option prefix
                    answer_match = re.search(r'([A-D])\.', response)
                    if answer_match:
                        answer = answer_match.group(1)
                    else:
                        raise ValueError("No valid answer found in response")
            
            # try to match explanation
            explanation_match = re.search(r'Explanation:\s*(.*?)(?=\n|$)', response, re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else ""
            
            return {
                'answer': answer,
                'explanation': explanation
            }
        except Exception as e:
            print(f"Error parsing answer: {str(e)}")
            print(f"Response: {response}")
            return {
                'answer': '',
                'explanation': str(e)
            }

    def run_benchmark(self, output_path: str):
        """Run benchmark on test set"""
        results = []
        total = len(self.test_set)
        
        # adjust batch size based on setting type
        if self.setting == "base":
            batch_size = 16
        else:  # mix and golden settings use smaller batch size
            batch_size = 8 # changed to 4, because each prompt is larger
        
        print(f"\nTotal questions: {total}")
        print(f"Batch size: {batch_size}")
        print(f"Number of batches: {(total + batch_size - 1) // batch_size}\n")
        
        # process by batches
        for i in range(0, total, batch_size):
            batch = self.test_set[i:i + batch_size]
            current_batch = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size
            print(f"\nProcessing batch {current_batch}/{total_batches} (questions {i+1}-{min(i+batch_size, total)}/{total})...")
            
            # print current GPU memory status
            print("\nCurrent GPU Memory Status:")
            print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
            print(f"Cached Memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")
            
            try:
                # for mix setting, filter out questions already in golden lookup table
                if self.setting == "mix":
                    filtered_batch = []
                    for item in batch:
                        question = item['question']
                        passage = item.get('passage', '')
                        lookup_key = f"{question}_{passage}"
                        if lookup_key not in self.golden_lookup:
                            filtered_batch.append(item)
                    
                    if filtered_batch:
                        # prepare batch data (only include questions that need model inference)
                        batch_queries = [item['question'] for item in filtered_batch]
                        batch_contexts = [item['context'] for item in filtered_batch]
                        batch_options = [item['options'] for item in filtered_batch]
                        
                        # generate answers
                        batch_responses = self.generate_answer_batch(
                            queries=batch_queries,
                            contexts=batch_contexts,
                            options_list=batch_options
                        )
                    else:
                        # all questions are in golden lookup table, no model inference needed
                        batch_responses = []
                        filtered_batch = []
                else:
                    # non-mix setting, normal processing
                    filtered_batch = batch
                    batch_queries = [item['question'] for item in batch]
                    batch_contexts = [item['context'] for item in batch]
                    batch_options = [item['options'] for item in batch]
                    
                    # generate answers
                    batch_responses = self.generate_answer_batch(
                        queries=batch_queries,
                        contexts=batch_contexts,
                        options_list=batch_options
                    )
                
                # process each answer (including from golden lookup table and model generated)
                processed_items = set()
                
                # process model generated results
                for item, response in zip(filtered_batch, batch_responses):
                    try:
                            # check if result can be obtained from golden lookup table
                        question = item['question']
                        passage = item.get('passage', '')
                        lookup_key = f"{question}_{passage}"
                        
                        if self.setting == "mix" and lookup_key in self.golden_lookup:
                            # directly use golden result
                            golden_result = self.golden_lookup[lookup_key]
                            result = {
                                'question': item['question'],
                                'options': item['options'],
                                'correct_answer': item['correct_answer'],
                                'model_answer': golden_result['model_answer'],
                                'model_explanation': golden_result['model_explanation'],
                                'is_correct': golden_result['is_correct']
                            }
                            
                            # ensure passage field is correctly passed to the result
                            if 'passage' in item:
                                result['passage'] = item['passage']
                                
                            print(f"✅ Found in golden lookup: {item['question'][:50]}...")
                        else:
                            # parse model generated answer
                            parsed_answer = self.parse_answer(response)
                            
                            # verify answer
                            is_correct = parsed_answer['answer'] == item['correct_answer']
                            
                            # save result
                            result = {
                                'question': item['question'],
                                'options': item['options'],
                                'correct_answer': item['correct_answer'],
                                'model_answer': parsed_answer['answer'],
                                'model_explanation': parsed_answer['explanation'],
                                'is_correct': is_correct
                            }
                            
                            # ensure passage field is correctly passed to the result
                            if 'passage' in item:
                                result['passage'] = item['passage']
                        
                        results.append(result)
                        
                        # print progress
                        if self.setting == "mix" and lookup_key in self.golden_lookup:
                            print(f"Question: {item['question'][:100]}... (from golden)")
                            print(f"Correct answer: {item['correct_answer']}")
                            print(f"Model answer: {result['model_answer']}")
                            print(f"Correct: {result['is_correct']}\n")
                        else:
                            print(f"Question: {item['question'][:100]}...")
                            print(f"Correct answer: {item['correct_answer']}")
                            print(f"Model answer: {result['model_answer']}")
                            print(f"Correct: {result['is_correct']}\n")
                        
                        processed_items.add(lookup_key)
                        
                    except Exception as e:
                        print(f"Error processing answer: {str(e)}")
                        error_result = {
                            'question': item['question'],
                            'options': item['options'],
                            'correct_answer': item['correct_answer'],
                            'model_answer': '',
                            'model_explanation': str(e),
                            'is_correct': False
                        }
                        
                        # ensure passage field is correctly passed to the result
                        if 'passage' in item:
                            error_result['passage'] = item['passage']
                            
                        results.append(error_result)
                        processed_items.add(lookup_key)
                
                # process questions in golden lookup table but not in filtered_batch
                if self.setting == "mix":
                    for item in batch:
                        question = item['question']
                        passage = item.get('passage', '')
                        lookup_key = f"{question}_{passage}"
                        if lookup_key not in processed_items and lookup_key in self.golden_lookup:
                            golden_result = self.golden_lookup[lookup_key]
                            result = {
                                'question': item['question'],
                                'options': item['options'],
                                'correct_answer': item['correct_answer'],
                                'model_answer': golden_result['model_answer'],
                                'model_explanation': golden_result['model_explanation'],
                                'is_correct': golden_result['is_correct']
                            }
                            
                            # ensure passage field is correctly passed to the result
                            if 'passage' in item:
                                result['passage'] = item['passage']
                                
                            results.append(result)
                            print(f"✅ Added from golden lookup: {item['question'][:50]}...")
                
                # clear cache
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                # if batch processing fails, fall back to single processing
                for item in batch:
                    try:
                        response = self.generate_answer(
                            query=item['question'],
                            context=item.get('context', []),
                            options=item['options'],
                            five_passages=item.get('five_passages')
                        )
                        parsed_answer = self.parse_answer(response)
                        is_correct = parsed_answer['answer'] == item['correct_answer']
                        single_result = {
                            'question': item['question'],
                            'options': item['options'],
                            'correct_answer': item['correct_answer'],
                            'model_answer': parsed_answer['answer'],
                            'model_explanation': parsed_answer['explanation'],
                            'is_correct': is_correct
                        }
                        
                        # ensure passage field is correctly passed to the result
                        if 'passage' in item:
                            single_result['passage'] = item['passage']
                            
                        results.append(single_result)
                    except Exception as e:
                        print(f"Error processing single item: {str(e)}")
                        error_result = {
                            'question': item['question'],
                            'options': item['options'],
                            'correct_answer': item['correct_answer'],
                            'model_answer': '',
                            'model_explanation': str(e),
                            'is_correct': False
                        }
                        
                        # ensure passage field is correctly passed to the result
                        if 'passage' in item:
                            error_result['passage'] = item['passage']
                            
                        results.append(error_result)
                
                # clear cache
                torch.cuda.empty_cache()
        
        # save results
        print("\nSaving results...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("Results saved successfully!")
        
        # calculate statistics
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r['is_correct'])
        accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        
        print(f"\nResults for {self.model_name} ({self.setting}):")
        print(f"Total questions: {total_questions}")
        print(f"Correct answers: {correct_answers}")
        print(f"Accuracy: {accuracy:.2f}%")

    def generate_answer_batch(self, queries: List[str], contexts: List[List[str]], options_list: List[List[str]]) -> List[str]:
        """Generate answers for a batch of questions"""
        try:
            # check if small model
            is_small_model = self.model_config.get('is_small_model', False)
            
            # prepare batch prompts
            batch_prompts = []
            for query, context, options in zip(queries, contexts, options_list):
                # truncate context for small model
                context = self._truncate_context_for_small_model(context)
                # remove option prefix
                clean_options = [opt.split('. ', 1)[1] for opt in options]
                options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(clean_options)])
                context_str = "\n".join(context) if context else ""
                # uniform 0-shot prompt
                if context_str:
                    prompt = f"""Passage: {context_str}\n\nQuestion: {query}\nOptions:\n{options_str}\n\nAnswer:"""
                else:
                    prompt = f"""Question: {query}\nOptions:\n{options_str}\n\nAnswer:"""
                batch_prompts.append(prompt)
            
            # prepare input
            print("\nTokenizing batch input...")
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model_config['max_sequence_length']
            ).to(self.model.device)
            
            # remove token_type_ids to avoid compatibility issues with some models
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            
            print(f"Batch input shape: {inputs['input_ids'].shape}")
            
            # generate answers
            print("\nGenerating batch responses...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.model_config["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # decode output
            print("Decoding batch responses...")
            responses = []
            for i, output in enumerate(outputs):
                response = self.tokenizer.decode(output, skip_special_tokens=True)
                # remove prompt
                response = response[len(batch_prompts[i]):].strip()
                responses.append(response)
            
            return responses
            
        except Exception as e:
            print(f"Error in batch generation: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Run benchmark for local models')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode (first 10 items only)')
    args = parser.parse_args()

    # Get base directory (project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    test_sets = [
        os.path.join(base_dir, "DATA", "final_mcq", "base_2000.json"),
        os.path.join(base_dir, "DATA", "final_mcq", "golden_2000.json"),
        os.path.join(base_dir, "DATA", "final_mcq", "mix_2000.json")
    ]

    # all models to evaluate
    model_names = list(MODEL_CONFIGS.keys())

    for model_name in model_names:
        print(f"\nProcessing model: {model_name}")
        print(f"Loading model {model_name} from {MODEL_CONFIGS[model_name]['path']}...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_CONFIGS[model_name]['path'],
                trust_remote_code=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_CONFIGS[model_name]['path'],
                torch_dtype=MODEL_CONFIGS[model_name]['torch_dtype'],
                device_map=MODEL_CONFIGS[model_name]['device_map'],
                trust_remote_code=True
            )
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            print(f"Failed to load model {model_name}, skipping...")
            continue

        for test_set in test_sets:
            setting = os.path.basename(test_set).split('_')[0]  # base/golden/mix
            output_path = f"{base_dir}/DATA/local_MCQ/{model_name}/{setting}_test.json"
            if os.path.exists(output_path):
                print(f"\nOutput file for {setting} setting already exists at: {output_path}")
                print("Skipping this setting...")
                continue
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            print(f"\nProcessing {setting} setting...")
            evaluator = LocalModelEvaluator(test_set, model_name, model, tokenizer)
            if args.test_mode:
                evaluator.test_set = evaluator.test_set[:10]
                print("Running in test mode (first 10 items only)")
            evaluator.run_benchmark(output_path)

if __name__ == "__main__":
    main() 