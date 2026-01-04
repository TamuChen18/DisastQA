import json
import os
import argparse
from typing import List, Dict, Any
import openai
from tqdm import tqdm
import requests
import time
import re
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random
import asyncio
import aiohttp

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

# Closed-source model configuration
CLOSED_MODEL_CONFIGS = {
    "gpt-4o": {
        "temperature": 0.7,
        "max_tokens": 500,
        "model_type": "openai"
    },
    "gpt-4o-mini": {
        "temperature": 0.7,
        "max_tokens": 500,
        "model_type": "openai"
    },
    "gpt-3.5-turbo": {
        "temperature": 0.7,
        "max_tokens": 500,
        "model_type": "openai"
    },
    "gemini-2.5-pro": {
        "temperature": 0.7,
        "max_tokens": 500,
        "model_type": "google"
    },
    "gemini-2.5-flash": {
        "temperature": 0.7,
        "max_tokens": 500,
        "model_type": "google"
    },
    "gemini-1.5-pro": {
        "temperature": 0.7,
        "max_tokens": 500,
        "model_type": "google"
    },
    "gemini-1.5-flash": {
        "temperature": 0.7,
        "max_tokens": 500,
        "model_type": "google"
    },
    "gemini-3-pro": {
        "temperature": 0.7,
        "max_tokens": 500,
        "model_type": "google"
    },
    "gpt-5.2": {
        "temperature": 0.7,
        "max_tokens": 500,
        "model_type": "openai"
    }
}

class ClosedModelOEEvaluator:
    """Evaluator for closed models (GPT, Gemini) on Open-Ended questions with difficulty-aware generation"""
    
    def __init__(self, test_set_path: str, model_name: str):
        """Initialize evaluator"""
        # Load .env file from DATA directory
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        load_dotenv(dotenv_path=env_path)
        
        self.test_set_path = test_set_path
        self.model_name = model_name
        self.model_config = CLOSED_MODEL_CONFIGS[model_name]
        self.setting = os.path.basename(test_set_path).split('_')[0]  # base/golden/mix
        
        # Load passages_by_score data for mix setting (to construct 5 passages)
        self.passages_data = {}
        if self.setting == "mix":
            self._load_passages_data()
        
        # API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Thread safety
        self.print_lock = Lock()
        self.results_lock = Lock()
        
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

    def _get_difficulty_config(self, difficulty: str) -> Dict[str, Any]:
        """Get difficulty-specific configuration"""
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

    async def generate_answer(self, query: str, context: List[str] = None, difficulty: str = "medium", five_passages: List[str] = None) -> str:
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
                
                # Call the appropriate model
                response = await self._call_model(prompt)
                
                # Clean up the response
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
                await asyncio.sleep(1)  # Wait for 1 second before retrying

    async def _call_model(self, prompt: str) -> str:
        """Call the appropriate model based on model type."""
        if self.model_config["model_type"] == "openai":
            return await self._call_openai(prompt)
        elif self.model_config["model_type"] == "google":
            return await self._call_google(prompt)
        else:
            raise ValueError(f"Unknown model type: {self.model_config['model_type']}")

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API asynchronously."""
        client = openai.AsyncOpenAI(api_key=self.openai_api_key)
        try:
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides detailed answers to open-ended questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.model_config["temperature"],
                max_tokens=self.model_config["max_tokens"]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API for model {self.model_name}: {str(e)}")
            raise

    async def _call_google(self, prompt: str) -> str:
        """Call Google Gemini API using standard REST API."""
        max_retries = 5
        base = 1.0  # Start with 1 second base delay
        
        for attempt in range(max_retries):
            try:
                # Use standard Generative Language API
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
                headers = {"Content-Type": "application/json"}
                params = {"key": self.google_api_key}
                
                data = {
                    "systemInstruction": {
                        "parts": [{"text": "You are a helpful assistant that provides detailed answers to open-ended questions."}]
                    },
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }],
                    "generationConfig": {
                        "temperature": self.model_config["temperature"],
                        "maxOutputTokens": self.model_config["max_tokens"]
                    }
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, params=params, json=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result["candidates"][0]["content"]["parts"][0]["text"]
                        else:
                            error_text = await response.text()
                            print(f"Google API error (attempt {attempt + 1}): {error_text}")
                            
                            if "quota" in error_text.lower() or "rate" in error_text.lower():
                                # Rate limit or quota exceeded
                                wait_time = base * (2 ** attempt)
                                print(f"Rate limited. Waiting {wait_time} seconds...")
                                await asyncio.sleep(wait_time)
                            else:
                                # Other error, don't retry
                                break
                                
            except Exception as e:
                print(f"Error calling Google API (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(base * (2 ** attempt))

    async def run_benchmark(self, output_path: str, concurrency: int = 500):
        """Run benchmark with high concurrency processing"""
        results = []
        total = len(self.test_set)
        
        print(f"\nTotal questions: {total}")
        print(f"Processing model: {self.model_name}")
        
        # Set concurrency number
        print(f"Concurrency: {concurrency}")
        
        # Create semaphore to control concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_item_with_semaphore(item):
            """Process single item with semaphore"""
            async with semaphore:
                try:
                    # Generate answer
                    print(f"🔄 Generating answer for: {item['question'][:50]}...")
                    response = await self.generate_answer(
                        query=item['question'],
                        context=item.get('context', []),
                        difficulty=item['difficulty'],
                        five_passages=item.get('five_passages')
                    )
                    
                    return {
                        'question': item['question'],
                        'correct_answer': item['correct_answer'],
                        'model_answer': response,
                        'answer_coverage': None,
                        'difficulty': item['difficulty'],
                        'passage': item.get('passage', '')
                    }
                    
                except Exception as e:
                    print(f"❌ Error processing item: {str(e)}")
                    return {
                        'question': item['question'],
                        'correct_answer': item['correct_answer'],
                        'model_answer': '',
                        'answer_coverage': None,
                        'difficulty': item['difficulty'],
                        'passage': item.get('passage', '')
                    }
        
        # Create all tasks
        print(f"\nCreating {total} tasks for concurrent processing...")
        tasks = [process_item_with_semaphore(item) for item in self.test_set]
        
        # Execute all tasks concurrently
        print("Starting concurrent processing...")
        start_time = time.time()
        
        # Execute all tasks concurrently using asyncio.gather
        processed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        print(f"Concurrent processing completed in {end_time - start_time:.2f} seconds")
        
        # Process results
        for i, result in enumerate(processed_results):
            if isinstance(result, Exception):
                print(f"❌ Task {i} failed with exception: {result}")
                # Create error result
                item = self.test_set[i]
                results.append({
                    'question': item['question'],
                    'correct_answer': item['correct_answer'],
                    'model_answer': '',
                    'answer_coverage': None,
                    'difficulty': item['difficulty'],
                    'passage': item.get('passage', '')
                })
            else:
                results.append(result)
        
        # Save result
        print("\nSaving results...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("Results saved successfully!")
        
        # Calculate statistics
        total_questions = len(results)
        valid_answers = sum(1 for r in results if r['model_answer'].strip())
        
        print(f"\nResults for {self.model_name} ({self.setting}):")
        print(f"Total questions: {total_questions}")
        print(f"Valid answers: {valid_answers}")
        print(f"Response rate: {(valid_answers / total_questions) * 100:.2f}%")
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        print(f"Average time per question: {(end_time - start_time) / total_questions:.2f} seconds")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run benchmark for closed models on OE questions with difficulty-aware generation')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode (first 10 items only)')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., gpt-4o, gemini-2.5-pro)')
    parser.add_argument('--concurrency', type=int, help='Concurrency level (auto-determined if not specified)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    # Change to OE test set path, use files with difficulty
    test_sets = [
        os.path.join(base_dir, "DATA", "final_OE", "base_oe_with_difficulty.json"),
        os.path.join(base_dir, "DATA", "final_OE", "golden_oe_with_difficulty.json"),
        os.path.join(base_dir, "DATA", "final_OE", "mix_oe_with_difficulty.json")
    ]

    model_name = args.model
    
    if model_name not in CLOSED_MODEL_CONFIGS:
        print(f"Error: Model {model_name} not supported. Supported models: {list(CLOSED_MODEL_CONFIGS.keys())}")
        return

    # Auto set concurrency number based on model
    if args.concurrency is None:
        concurrency = 500 if model_name.startswith("gemini") else 3000  # GPT uses 3000, Gemini uses 500
        print(f"Auto-setting concurrency: {concurrency} for {model_name}")
    else:
        concurrency = args.concurrency
        print(f"Using specified concurrency: {concurrency}")

    print(f"\nProcessing model: {model_name}")
    
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
        
        evaluator = ClosedModelOEEvaluator(test_set, model_name)
        
        if args.test_mode:
            evaluator.test_set = evaluator.test_set[:10]
            print("Running in test mode (first 10 items only)")
        
        await evaluator.run_benchmark(output_path, concurrency)

if __name__ == "__main__":
    asyncio.run(main())
