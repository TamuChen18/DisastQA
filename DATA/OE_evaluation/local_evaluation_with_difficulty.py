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

# Difficulty-adjusted token configuration - optimized based on keypoint coverage analysis
DIFFICULTY_TOKEN_CONFIG = {
    "easy": {
        "max_new_tokens": 80,  # Easy questions, short answers (100% coverage, no adjustment needed)
        "word_limit": "10-100"
    },
    "medium": {
        "max_new_tokens": 180,  # Medium questions, medium-length answers (100% coverage, no adjustment needed)
        "word_limit": "20-200"
    },
    "hard": {
        "max_new_tokens": 300,  # Hard questions, increase tokens to improve coverage (current 66.7% coverage)
        "word_limit": "60-350"  # Increase length range to cover more keypoints
    },
    "extremely_complex": {
        "max_new_tokens": 400,  # Extremely complex questions, significantly increase tokens (current 66.7% coverage)
        "word_limit": "120-500"  # Increase length range to cover more keypoints
    }
}

# Model configuration - for OE question answering
MODEL_CONFIGS = {
    "llama-3-8b": {
        "path": os.path.join(_BASE_DIR, "models", "llama-3-8b"),
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "base_max_new_tokens": 128,  # Base token count, will be adjusted based on difficulty
        "generation_config": {
            "temperature": 0.7,   # Moderate temperature, balancing creativity and accuracy
            "do_sample": True,    # Enable sampling
        }
    },
    "llama-3.2-3b-instruct": {
        "path": os.path.join(_BASE_DIR, "models", "llama-3.2-3b-instruct"),
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "base_max_new_tokens": 100,
        "generation_config": {
            "temperature": 0.7,   # Moderate temperature
            "do_sample": True,    # Enable sampling
        }
    },
    "Mistral-7B-Instruct-v0.2": {
        "path": os.path.join(_BASE_DIR, "models", "Mistral-7B-Instruct-v0.2"),
        "max_sequence_length": 8192,  # 32K max, using 8K as safe value
        "torch_dtype": torch.float16,
        "base_max_new_tokens": 128,  # Base token count, will be adjusted based on difficulty
        "device_map": "auto",
        "generation_config": {
            "temperature": 0.7,   # Moderate temperature, balancing creativity and accuracy
            "do_sample": True,    # Enable sampling
        }
    },
    "qwen-2.5-3b-instruct": {
        "path": os.path.join(_BASE_DIR, "models", "qwen-2.5-3b-instruct"),
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "base_max_new_tokens": 100,
        "generation_config": {
            "temperature": 0.7,
            "do_sample": True,
        }
    },
    "qwen-3-4b": {
        "path": os.path.join(_BASE_DIR, "models", "qwen-3-4b"),
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "base_max_new_tokens": 100,
        "generation_config": {
            "temperature": 0.7,
            "do_sample": True,
        }
    },
    "qwen-3-8b": {
        "path": os.path.join(_BASE_DIR, "models", "qwen-3-8b"),
        "base_max_new_tokens": 128,  # Base token count, will be adjusted based on difficulty
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "generation_config": {
            "temperature": 0.7,   # Moderate temperature
            "do_sample": True,    # Enable sampling
        }
    },
    "deepseek-v3-7b": {
        "path": os.path.join(_BASE_DIR, "models", "deepseek-v3-7b"),
        "base_max_new_tokens": 128,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "generation_config": {
            "temperature": 0.7,    # Moderate temperature
            "do_sample": True,     # Enable sampling
        }
    },
    "phi-2": {
        "path": os.path.join(_BASE_DIR, "models", "phi-2"),
        "base_max_new_tokens": 150,  # Small model, reduce token count
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 2048,  # Actual limit is 2K
        "generation_config": {
            "temperature": 0.7,    # Moderate temperature
            "do_sample": True,     # Enable sampling
        }
    },
    "gemma-7b": {
        "path": os.path.join(_BASE_DIR, "models", "gemma-7b"),
        "base_max_new_tokens": 100,
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 2048,
        "generation_config": {
            "temperature": 0.7,    # Moderate temperature
            "do_sample": True,     # Enable sampling
        }
    },
    "Llama-3.2-1B-Instruct": {
        "path": os.path.join(_BASE_DIR, "models", "Llama-3.2-1B-Instruct"),
        "torch_dtype": torch.float16,
        "base_max_new_tokens": 150,  # Small model
        "device_map": "auto",
        "max_sequence_length": 2048,
        "generation_config": {
            "temperature": 0.7,
            "do_sample": True,
        }
    },
    "qwen-3-0.6b": {
        "path": os.path.join(_BASE_DIR, "models", "qwen-3-0.6b"),
        "torch_dtype": torch.float16,
        "base_max_new_tokens": 150,  # Small model
        "device_map": "auto",
        "max_sequence_length": 8192,  # 40K max, using 8K as safe value
        "generation_config": {
            "temperature": 0.7,
            "do_sample": True,
        }
    },
    "Hunyuan-7B-Instruct": {
        "path": os.path.join(_BASE_DIR, "models", "Hunyuan-7B-Instruct"),
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
        "max_sequence_length": 8192,
        "base_max_new_tokens": 128,
        "generation_config": {
            "temperature": 0.7,
            "do_sample": True,
        }
    },
    "Hunyuan-4B-Instruct": {
        "path": os.path.join(_BASE_DIR, "models", "Hunyuan-4B-Instruct"),
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
        "max_sequence_length": 8192,
        "base_max_new_tokens": 200,
        "generation_config": {
            "temperature": 0.7,
            "do_sample": True,
        }
    },
    "Yi-6B-Chat": {
        "path": os.path.join(_BASE_DIR, "models", "Yi-6B-Chat"),
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 8192,
        "base_max_new_tokens": 128,
        "generation_config": {
            "temperature": 0.7,
            "do_sample": True,
        }
    },
    "AceMath-1.5B-Instruct": {
        "path": os.path.join(_BASE_DIR, "models", "AceMath-1.5B-Instruct"),
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 2048,
        "base_max_new_tokens": 150,
        "generation_config": {
            "temperature": 0.7,
            "do_sample": True,
        }
    },
    "Falcon3-1B-Instruct": {
        "path": os.path.join(_BASE_DIR, "models", "Falcon3-1B-Instruct"),
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "max_sequence_length": 2048,
        "base_max_new_tokens": 150,
        "generation_config": {
            "temperature": 0.7,
            "do_sample": True,
        }
    }
}

class LocalOEModelEvaluator:
    """Evaluator for local models on Open-Ended questions with difficulty-aware generation"""
    
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
        
        # Print initial GPU memory status
        print("\nInitial GPU Memory Status:")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")
        print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
        print(f"Cached Memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")
        
        # Load test set
        with open(test_set_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Convert data format for OE with difficulty
        self.test_set = []
        for item in raw_data:
            if 'open_ended' in item and 'gpt40' in item['open_ended']:
                content = item['open_ended']['gpt40']['content']
                passage = item.get('passage', '')  # Save original passage (for golden/base)
                original_query = item.get('original_query', '')  # For mix setting to load passages_by_score
                difficulty = item.get('difficulty', 'medium')  # Get difficulty, default to medium
                
                # For mix setting, construct 5 passages (1 golden + 4 distractors)
                if self.setting == "mix" and original_query:
                    five_passages = self._construct_five_passages(original_query)
                else:
                    five_passages = None
                
                processed_item = {
                    'question': content['question'],
                    'correct_answer': content['correct_answer'],  # OE correct answer
                    'llm_answer': content.get('llm_answer', ''),  # OE LLM answer
                    'context': [passage] if self.setting in ['golden'] and passage else [],  # Single passage for golden
                    'passage': passage,  # Save passage for reference
                    'five_passages': five_passages,  # 5 passages for mix setting
                    'original_query': original_query,  # Save original_query for mix
                    'difficulty': difficulty  # Add difficulty field
                }
                self.test_set.append(processed_item)
        
        # Statistics difficulty distribution
        difficulty_counts = {}
        for item in self.test_set:
            difficulty = item['difficulty']
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        print(f"\nDifficulty distribution in test set:")
        for difficulty, count in difficulty_counts.items():
            print(f"  {difficulty}: {count} questions")
        
        # Verify data format
        required_keys = ['question', 'correct_answer', 'difficulty']
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
            # Set padding token and padding side
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'  # Set padding side to left
            
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
            # Set padding token and padding side
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'  # Set padding side to left
        
        # Initialize generation config
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
            data_prepare_dir = f"{base_dir}/DATA/DATA/data_prepare"
            
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
    
    def _get_difficulty_config(self, difficulty: str) -> Dict[str, Any]:
        """Get corresponding configuration based on difficulty"""
        return DIFFICULTY_TOKEN_CONFIG.get(difficulty, DIFFICULTY_TOKEN_CONFIG["medium"])
    
    def _clean_response(self, response: str) -> str:
        """Clean response, remove format instructions and irrelevant content, and control length"""
        if not response:
            return response
        
        # Remove common format instructions
        format_patterns = [
            r'The answer should be in English[^.]*\.',
            r'Please do not use any markdown[^.]*\.',
            r'Please ensure that the answer is[^.]*\.',
            r'Please type the answer here[^.]*\.',
            r'Answer:\s*\([^)]*\)',  # Remove "Answer: (Please type...)"
            r'Okay, so[^.]*\.',  # Remove "Okay, so..."
            r'Let me[^.]*\.',  # Remove "Let me..."
            r'I will[^.]*\.',  # Remove "I will..."
            r'Based on[^.]*\.',  # Remove "Based on..."
            r'\(Source:[^)]*\)',  # Remove "(Source: ...)"
            r'\(Word Count:[^)]*\)',  # Remove "(Word Count: ...)"
            r'\(Time[^)]*\)',  # Remove "(Time...)"
            r'\[[0-9]+\]',  # Remove "[1] [2] [3]" citation markers
        ]
        
        cleaned_response = response
        for pattern in format_patterns:
            cleaned_response = re.sub(pattern, '', cleaned_response, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove extra blank lines and spaces
        cleaned_response = re.sub(r'\n\s*\n', '\n', cleaned_response)
        cleaned_response = re.sub(r'^\s+|\s+$', '', cleaned_response, flags=re.MULTILINE)
        
        # If cleaned result is empty, return original response
        if not cleaned_response.strip():
            return response.strip()
        
        # Record length information, but do not truncate
        words = cleaned_response.split()
        if len(words) > 500:  # Increase warning threshold to accommodate new extremely_complex configuration
            print(f"⚠️  Warning: Answer is quite long ({len(words)} words). Consider adjusting prompt or token limits.")
        
        return cleaned_response.strip()
    
    def _truncate_context_for_small_model(self, context: List[str], max_tokens_per_passage: int = 200) -> List[str]:
        """Truncate context to fit small model token limits"""
        if not self.model_config.get('is_small_model', False):
            return context
        
        truncated_context = []
        for passage in context:
            # Simple character-based truncation, approximately 4 characters = 1 token
            if len(passage) > max_tokens_per_passage * 4:
                passage = passage[:max_tokens_per_passage * 4] + "..."
            truncated_context.append(passage)
        
        return truncated_context

    def generate_answer(self, query: str, context: List[str] = None, difficulty: str = "medium", five_passages: List[str] = None) -> str:
        """Generate answer for an open-ended question with difficulty-aware generation"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Get difficulty config
                difficulty_config = self._get_difficulty_config(difficulty)
                word_limit = difficulty_config["word_limit"]
                
                # Build prompt based on setting
                if five_passages and len(five_passages) == 5:
                    # Mix setting: 5 passages with selection instruction
                    passages_text = "\n\n".join([f"Passage {i+1}: {passage}" for i, passage in enumerate(five_passages)])
                    prompt = f"""You are given 5 passages (some may be irrelevant). You must select ONLY ONE passage that is most relevant to answer the question. First output the passage number you selected in the format "Passage: <single number between 1 and 5>" (only one number, no commas or multiple numbers), then provide a comprehensive answer based on that passage.

{passages_text}

Question: {query}
Difficulty Level: {difficulty.upper()}

Instructions: Provide a comprehensive answer within {word_limit} words. Ensure you cover all important aspects and key points related to the question. Be thorough but concise.

Answer:"""
                elif context:
                    # Golden setting: single passage
                    context = self._truncate_context_for_small_model(context)
                    context_str = "\n".join(context) if isinstance(context, list) else context
                    prompt = f"""Passage: {context_str}

Question: {query}
Difficulty Level: {difficulty.upper()}

Instructions: Provide a comprehensive answer within {word_limit} words. Ensure you cover all important aspects and key points related to the question. Be thorough but concise.

Answer:"""
                else:
                    # Base setting: no passage
                    prompt = f"""Question: {query}

Difficulty Level: {difficulty.upper()}

Instructions: Provide a comprehensive answer within {word_limit} words. Ensure you cover all important aspects and key points related to the question. Be thorough but concise.

Answer:"""
                
                print(f"\nGenerating response for {difficulty} difficulty...")
                print(f"Prompt length: {len(prompt)}")
                
                # Generate response
                print("Tokenizing input...")
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.model_config['max_sequence_length']
                ).to(self.model.device)
                
                # Remove token_type_ids to avoid compatibility issues with certain models
                if 'token_type_ids' in inputs:
                    del inputs['token_type_ids']
                
                print(f"Input length: {len(inputs['input_ids'][0])}")
                print("Starting generation...")
                
                # Adjust max_new_tokens based on difficulty
                base_tokens = self.model_config["base_max_new_tokens"]
                difficulty_tokens = difficulty_config["max_new_tokens"]
                # Prefer difficulty configuration, but don't exceed model capability limit
                adjusted_tokens = min(difficulty_tokens, 500)  # Set 500 as safe upper limit
                
                print(f"Using {adjusted_tokens} tokens for {difficulty} difficulty")
                
                # Generate with model-specific parameters
                with torch.no_grad():
                    try:
                        # Add progress indicator
                        print("Generating tokens...", end="", flush=True)
                        start_time = time.time()
                        # Use generation parameters from model configuration
                        gen_config = self.model_config["generation_config"]
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=adjusted_tokens,
                            temperature=gen_config.temperature if hasattr(gen_config, 'temperature') else gen_config.get('temperature', 0.7),
                            do_sample=gen_config.do_sample if hasattr(gen_config, 'do_sample') else gen_config.get('do_sample', True),
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
                
                # Clean up the response to remove format instructions and improve quality
                response = self._clean_response(response)
                
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

    def parse_answer(self, response: str) -> str:
        """Parse model's answer from response for OE questions"""
        try:
            # For OE questions, directly return entire response as answer
            # Try to match content after "Answer:"
            answer_match = re.search(r'Answer:\s*(.*?)(?=\n|$)', response, re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()
            else:
                # If "Answer:" marker not found, directly use entire response
                answer = response.strip()
            
            return answer
        except Exception as e:
            print(f"Error parsing answer: {str(e)}")
            print(f"Response: {response}")
            return ""

    def run_benchmark(self, output_path: str):
        """Run benchmark on test set with difficulty-aware processing"""
        results = []
        total = len(self.test_set)
        
        # Adjust batch size based on setting type
        if self.setting == "base":
            batch_size = 16
        else:  # mix and golden settings use smaller batch sizes
            batch_size = 8 # Changed to 4 because each prompt will be larger
        
        print(f"\nTotal questions: {total}")
        print(f"Batch size: {batch_size}")
        print(f"Number of batches: {(total + batch_size - 1) // batch_size}\n")
        
        # Process by batch
        for i in range(0, total, batch_size):
            batch = self.test_set[i:i + batch_size]
            current_batch = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size
            print(f"\nProcessing batch {current_batch}/{total_batches} (questions {i+1}-{min(i+batch_size, total)}/{total})...")
            
            # Print current GPU memory status
            print("\nCurrent GPU Memory Status:")
            print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
            print(f"Cached Memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")
            
            try:
                # For mix setting, filter out questions already in golden lookup table
                if self.setting == "mix":
                    filtered_batch = []
                    for item in batch:
                        question = item['question']
                        passage = item.get('passage', '')
                        lookup_key = f"{question}_{passage}"
                        if lookup_key not in self.golden_lookup:
                            filtered_batch.append(item)
                    
                    if filtered_batch:
                        # Prepare batch data (only includes questions requiring model inference)
                        batch_queries = [item['question'] for item in filtered_batch]
                        batch_contexts = [item['context'] for item in filtered_batch]
                        batch_difficulties = [item['difficulty'] for item in filtered_batch]
                        
                        # Generate answer
                        batch_responses = self.generate_answer_batch(
                            queries=batch_queries,
                            contexts=batch_contexts,
                            difficulties=batch_difficulties
                        )
                    else:
                        # All questions are in golden lookup table, no model inference needed
                        batch_responses = []
                        filtered_batch = []
                else:
                    # Non-mix setting, normal processing
                    filtered_batch = batch
                    batch_queries = [item['question'] for item in batch]
                    batch_contexts = [item['context'] for item in batch]
                    batch_difficulties = [item['difficulty'] for item in batch]
                    
                    # Generate answer
                    batch_responses = self.generate_answer_batch(
                        queries=batch_queries,
                        contexts=batch_contexts,
                        difficulties=batch_difficulties
                    )
                
                # Process each answer (including those from golden lookup table and model-generated)
                processed_items = set()
                
                # Process model-generated results
                for item, response in zip(filtered_batch, batch_responses):
                    try:
                        # Check if can retrieve from golden lookup table
                        question = item['question']
                        passage = item.get('passage', '')
                        lookup_key = f"{question}_{passage}"
                        
                        if self.setting == "mix" and lookup_key in self.golden_lookup:
                            # Directly use golden result
                            golden_result = self.golden_lookup[lookup_key]
                            result = {
                                'question': item['question'],
                                'correct_answer': item['correct_answer'],
                                'model_answer': golden_result['model_answer'],
                                'answer_coverage': golden_result.get('answer_coverage', None),
                                'difficulty': item['difficulty']  # Add difficulty
                            }
                            
                            # Ensure passage field is correctly passed to results
                            if 'passage' in item:
                                result['passage'] = item['passage']
                                
                            print(f"✅ Found in golden lookup: {item['question'][:50]}...")
                        else:
                            # For OE questions, we save the generated answer without correctness judgment
                            result = {
                                'question': item['question'],
                                'correct_answer': item['correct_answer'],
                                'model_answer': response,
                                'answer_coverage': None,  # Will be calculated later through keypoint matching
                                'difficulty': item['difficulty']  # Add difficulty
                            }
                            
                            # Ensure passage field is correctly passed to results
                            if 'passage' in item:
                                result['passage'] = item['passage']
                        
                        results.append(result)
                        
                        # Print progress
                        if self.setting == "mix" and lookup_key in self.golden_lookup:
                            print(f"Question: {item['question'][:100]}... (from golden)")
                            print(f"Difficulty: {item['difficulty']}")
                            print(f"Correct answer: {item['correct_answer']}")
                            print(f"Model answer: {result['model_answer']}\n")
                        else:
                            print(f"Question: {item['question'][:100]}...")
                            print(f"Difficulty: {item['difficulty']}")
                            print(f"Correct answer: {item['correct_answer']}")
                            print(f"Model answer: {result['model_answer']}\n")
                        
                        processed_items.add(lookup_key)
                        
                    except Exception as e:
                        print(f"Error processing answer: {str(e)}")
                        error_result = {
                            'question': item['question'],
                            'correct_answer': item['correct_answer'],
                            'model_answer': '',
                            'answer_coverage': None,
                            'difficulty': item['difficulty']  # Add difficulty
                        }
                        
                        # Ensure passage field is correctly passed to results
                        if 'passage' in item:
                            error_result['passage'] = item['passage']
                            
                        results.append(error_result)
                        processed_items.add(lookup_key)
                
                # Process questions in golden lookup table but not in filtered_batch
                if self.setting == "mix":
                    for item in batch:
                        question = item['question']
                        passage = item.get('passage', '')
                        lookup_key = f"{question}_{passage}"
                        if lookup_key not in processed_items and lookup_key in self.golden_lookup:
                            golden_result = self.golden_lookup[lookup_key]
                            result = {
                                'question': item['question'],
                                'correct_answer': item['correct_answer'],
                                'model_answer': golden_result['model_answer'],
                                'answer_coverage': golden_result.get('answer_coverage', None),
                                'difficulty': item['difficulty']  # Add difficulty
                            }
                            
                            # Ensure passage field is correctly passed to results
                            if 'passage' in item:
                                result['passage'] = item['passage']
                                
                            results.append(result)
                            print(f"✅ Added from golden lookup: {item['question'][:50]}...")
                
                # Clear cache
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                # If batch processing fails, fall back to single processing
                for item in batch:
                    try:
                        response = self.generate_answer(
                            query=item['question'],
                            context=item.get('context', []),
                            difficulty=item['difficulty'],  # Pass difficulty
                            five_passages=item.get('five_passages')
                        )
                        single_result = {
                            'question': item['question'],
                            'correct_answer': item['correct_answer'],
                            'model_answer': response,
                            'answer_coverage': None,
                            'difficulty': item['difficulty']  # Add difficulty
                        }
                        
                        # Ensure passage field is correctly passed to results
                        if 'passage' in item:
                            single_result['passage'] = item['passage']
                            
                        results.append(single_result)
                    except Exception as e:
                        print(f"Error processing single item: {str(e)}")
                        error_result = {
                            'question': item['question'],
                            'correct_answer': item['correct_answer'],
                            'model_answer': '',
                            'answer_coverage': None,
                            'difficulty': item['difficulty']  # Add difficulty
                        }
                        
                        # Ensure passage field is correctly passed to results
                        if 'passage' in item:
                            error_result['passage'] = item['passage']
                            
                        results.append(error_result)
                
                # Clear cache
                torch.cuda.empty_cache()
        
        # Save result
        print("\nSaving results...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("Results saved successfully!")
        
        # Calculate statistics
        total_questions = len(results)
        print(f"\nResults for {self.model_name} ({self.setting}):")
        print(f"Total questions: {total_questions}")
        
        # Statistics by difficulty
        difficulty_stats = {}
        for result in results:
            difficulty = result.get('difficulty', 'unknown')
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = 0
            difficulty_stats[difficulty] += 1
        
        print("Results by difficulty:")
        for difficulty, count in difficulty_stats.items():
            print(f"  {difficulty}: {count} questions")
        
        print("Note: Answer coverage will be calculated later using keypoint matching.")

    def generate_answer_batch(self, queries: List[str], contexts: List[List[str]], difficulties: List[str]) -> List[str]:
        """Generate answers for a batch of open-ended questions with difficulty-aware generation"""
        try:
            # Check if is small model
            is_small_model = self.model_config.get('is_small_model', False)
            
            # Prepare batch prompts
            batch_prompts = []
            for query, context, difficulty in zip(queries, contexts, difficulties):
                # Truncate context for small models
                context = self._truncate_context_for_small_model(context)
                context_str = "\n".join(context) if context else ""
                
                # Get difficulty configuration
                difficulty_config = self._get_difficulty_config(difficulty)
                word_limit = difficulty_config["word_limit"]
                
                # Unified as 0-shot prompt with difficulty-aware length constraint
                if context_str:
                    prompt = f"""Passage: {context_str}

Question: {query}

Difficulty Level: {difficulty.upper()}

Instructions: Provide a focused answer within {word_limit} words. Be concise and directly address the question without unnecessary details.

Answer:"""
                else:
                    prompt = f"""Question: {query}

Difficulty Level: {difficulty.upper()}

Instructions: Provide a focused answer within {word_limit} words. Be concise and directly address the question without unnecessary details.

Answer:"""
                batch_prompts.append(prompt)
            
            # Prepare input
            print("\nTokenizing batch input...")
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model_config['max_sequence_length']
            ).to(self.model.device)
            
            # Remove token_type_ids to avoid compatibility issues with certain models
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            
            print(f"Batch input shape: {inputs['input_ids'].shape}")
            
            # Generate answer - use maximum difficulty token count to ensure coverage of all cases
            print("\nGenerating batch responses...")
            with torch.no_grad():
                gen_config = self.model_config["generation_config"]
                
                # Use base token count because batch may have different difficulties
                base_tokens = self.model_config["base_max_new_tokens"]
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=base_tokens,
                    temperature=gen_config.temperature if hasattr(gen_config, 'temperature') else gen_config.get('temperature', 0.7),
                    do_sample=gen_config.do_sample if hasattr(gen_config, 'do_sample') else gen_config.get('do_sample', True),
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            print("Decoding batch responses...")
            responses = []
            for i, output in enumerate(outputs):
                response = self.tokenizer.decode(output, skip_special_tokens=True)
                # Remove prompt
                response = response[len(batch_prompts[i]):].strip()
                # Clean response
                response = self._clean_response(response)
                responses.append(response)
            
            return responses
            
        except Exception as e:
            print(f"Error in batch generation: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Run benchmark for local models on OE questions with difficulty-aware generation')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode (first 10 items only)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    # Change to OE test set path, use files with difficulty
    test_sets = [
        os.path.join(base_dir, "DATA", "final_OE", "base_oe_with_difficulty.json"),
        os.path.join(base_dir, "DATA", "final_OE", "golden_oe_with_difficulty.json"),
        os.path.join(base_dir, "DATA", "final_OE", "mix_oe_with_difficulty.json")
    ]

    # All models you want to evaluate
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
            # Change output path to OE path
            output_path = os.path.join(base_dir, "DATA", "local_OE", model_name, f"{setting}_oe_with_difficulty.json")
            if os.path.exists(output_path):
                print(f"\nOutput file for {setting} setting already exists at: {output_path}")
                print("Skipping this setting...")
                continue
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            print(f"\nProcessing {setting} setting...")
            evaluator = LocalOEModelEvaluator(test_set, model_name, model, tokenizer)
            if args.test_mode:
                evaluator.test_set = evaluator.test_set[:10]
                print("Running in test mode (first 10 items only)")
            evaluator.run_benchmark(output_path)

if __name__ == "__main__":
    main() 