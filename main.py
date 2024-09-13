import argparse
import json
import logging
import time
from typing import List, Dict, Any

import instructor
from openai import OpenAI
from anthropic import Anthropic
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize API clients with Instructor
openai_client = instructor.patch(OpenAI())
anthropic_client = instructor.patch(Anthropic())

class Message(BaseModel):
    role: str
    content: str

class Conversation(BaseModel):
    messages: List[Message]

class JudgeScore(BaseModel):
    accuracy: float = Field(..., ge=1, le=10)
    quality: float = Field(..., ge=1, le=10)
    coherence: float = Field(..., ge=1, le=10)
    relevance: float = Field(..., ge=1, le=10)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset using AI models")
    parser.add_argument("description", type=str, help="Description of the dataset")
    parser.add_argument("example", type=str, help="Example entry for the dataset")
    parser.add_argument("num_examples", type=int, help="Number of examples to generate")
    parser.add_argument("output_file", type=str, help="Output JSON file path")
    return parser.parse_args()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_example(description: str, example: str) -> Conversation:
    prompt = f"""
    Generate a high-quality example for a dataset based on the following description and example:

    Description: {description}

    Example: {example}

    Generate a new, unique example following the same format and adhering to the description.
    """
    
    response = anthropic_client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        temperature=0.7,
        system="You are an AI assistant specialized in generating high-quality, diverse dataset examples.",
        messages=[{"role": "user", "content": prompt}],
        response_model=Conversation
    )
    
    logger.info(f"Generated example: {response}")
    return response

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def judge_example(description: str, example: str, generated_example: Conversation, judge_model: str) -> JudgeScore:
    prompt = f"""
    Evaluate the quality of the following generated example for a dataset:

    Description: {description}

    Original Example: {example}

    Generated Example: {json.dumps(generated_example.dict())}

    Please score the generated example on the following criteria:
    1. Accuracy (1-10): How well does it match the original description?
    2. Quality (1-10): How well-formed and coherent is the example?
    3. Coherence (1-10): How well do the different parts of the example fit together?
    4. Relevance (1-10): How relevant is the example to the described dataset?

    Provide your evaluation as a JSON object with these four scores.
    """

    if judge_model.startswith("gpt-"):
        response = openai_client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": "You are an AI judge evaluating the quality of generated dataset examples."},
                {"role": "user", "content": prompt}
            ],
            response_model=JudgeScore
        )
    else:
        response = anthropic_client.messages.create(
            model=judge_model,
            max_tokens=150,
            temperature=0,
            system="You are an AI judge evaluating the quality of generated dataset examples.",
            messages=[{"role": "user", "content": prompt}],
            response_model=JudgeScore
        )
    
    logger.info(f"Scores from {judge_model}: {response}")
    return response

def calculate_final_score(scores: List[JudgeScore]) -> float:
    return sum(score.accuracy + score.quality + score.coherence + score.relevance for score in scores)

def main():
    args = parse_arguments()
    
    dataset = []
    few_shot_examples = []
    judge_models = ["gpt-4-0125-preview", "claude-3-sonnet-20240229", "claude-3-sonnet-20240229"]
    
    try:
        while len(dataset) < args.num_examples:
            generated_example = generate_example(args.description, args.example)
            
            scores = [judge_example(args.description, args.example, generated_example, model) for model in judge_models]
            final_score = calculate_final_score(scores)
            
            threshold = 0.95 * (40 * len(judge_models))  # 95% of max possible score
            
            if final_score >= threshold:
                dataset.append(generated_example)
                few_shot_examples.append(generated_example)
                logger.info(f"Added example to dataset. Current count: {len(dataset)}")
            else:
                logger.info(f"Example rejected. Score: {final_score}, Threshold: {threshold}")
            
            if len(few_shot_examples) > 5:
                few_shot_examples.pop(0)
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.info("Saving current progress to output file.")
    
    finally:
        with open(args.output_file, 'w') as f:
            for example in dataset:
                json.dump(example.dict(), f)
                f.write('\n')
        
        logger.info(f"Generated {len(dataset)} examples. Saved to {args.output_file}")

if __name__ == "__main__":
    main()

