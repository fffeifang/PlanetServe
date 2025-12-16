import json
import math
import sys
from typing import Any, Dict, Optional

import requests
from scipy.stats import chi2
from math import exp

def send_completion(
    prompt: str,
    max_tokens: int = 7,
    temperature: float = 0,
    url: str = "http://localhost:8080/v1/completions",
) -> Dict[str, Any]:
    """Send a completion request to the local LLM server."""
    headers = {"Content-Type": "application/json"}

    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": 100,
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}



def check_credibility(
    prompt: str,
    output: str,
    tokenize_url: str = "http://localhost:8080/tokenize",
    completion_url: str = "http://localhost:8080/v1/completions",
    eps: float = 0.00001,
) -> float:
    """
    Check the credibility of an output given a prompt using token logprobs.

    Args:
        prompt: The input prompt
        output: The output to check
        tokenize_url: URL for tokenization endpoint
        completion_url: URL for completions endpoint

    Returns:
        float: Credibility score between 0 and 1
    """
    # First, tokenize the output
    try:
        tokenize_response = requests.post(
            tokenize_url,
            headers={"Content-Type": "application/json"},
            json={"content": output, "with_pieces": True},
        )
        tokenize_response.raise_for_status()
        tokens = tokenize_response.json()["tokens"]
        print(tokens)

        if not tokens:
            print("No tokens")
            return 0.0

        # Initialize variables for tracking probabilities
        token_count = 0

        # For each position, get the logprobs of the next token
        current_text = prompt
        total_prob = []
        for i in range(len(tokens) - 1):  # -1
            completion_response = send_completion(
                prompt=current_text, max_tokens=1, temperature=0, url=completion_url
            )

            # Get logprobs for the next token
            logprobs = completion_response["choices"][0]["logprobs"]["content"][0][
                "top_logprobs"
            ]
            next_token = tokens[i]["id"]  # Get the next token's ID

            # Find matching logprob entry
            matching_logprob = None
            for logprob_entry in logprobs:
                if logprob_entry["id"] == next_token:
                    matching_logprob = logprob_entry["logprob"]
                    token_count += 1
                    total_prob.append(pow(2.71828, matching_logprob))
                    print(f"Token: {tokens[i]['piece']}, LogProb: {matching_logprob}")
                    break

            if matching_logprob is None:
                total_prob.append(
                    eps
                )  # epsilon is a very small value close to zero for easy calculation
                print(f"No matching logprob found for token: {tokens[i]['piece']}")

            # Update current_text for next iteration
            current_text += tokens[i]["piece"]

        # Calculate average logprob and convert to a score between 0 and 1
        if token_count == 0:
            return 0.0

        print("Individual probabilities:", total_prob)

        # Apply geometric mean using the average of logarithms for numerical stability
        # Calculate the geometric mean of probabilities
        n = len(total_prob)
        if n > 0:
            # Take the logarithm of each probability, average them, then exponentiate
            log_probs = [math.log(p) if p > 0 else math.log(eps) for p in total_prob]
            avg_neg_logp = -sum(log_probs) / len(log_probs)
            perplexity = math.exp(avg_neg_logp)
            cred_score = 1.0 / perplexity  
        else:
            cred_score = 0.0

        print(f"Number of probabilities: {n}")
        print(f"Perplexity: {perplexity}")

        # Return the geometric mean as our credibility score
        return cred_score

    except Exception as e:
        print(f"Error in credibility check: {str(e)}")
        return 0.0


def check_llama_health(url: str = "http://localhost:8080/health") -> dict:
    """Check if llama-server is ready to handle requests."""
    try:
        response = requests.get(url, timeout=5)
        data = response.json()

        if response.status_code == 200 and data.get("status") == "ok":
            return {"healthy": True, "message": "Server ready"}

        if response.status_code == 503:
            error_msg = data.get("error", {}).get("message", "Unknown error")
            return {"healthy": False, "message": f"Server loading: {error_msg}"}

        return {
            "healthy": False,
            "message": f"Unexpected response: {response.status_code}",
        }

    except (requests.ConnectionError, requests.Timeout):
        return {"healthy": False, "message": "Server not responding"}
    except Exception as e:
        return {"healthy": False, "message": f"Error: {str(e)}"}


if __name__ == "__main__":
    # First check if server is healthy
    health_result = check_llama_health()
    if not health_result["healthy"]:
        print(f"Server health check failed: {health_result['message']}")
        sys.exit(1)

 

    # Read prompt and output from JSON files
    try:
        with open("../prompts.json", "r") as f:
            prompts_data = json.load(f)
        with open("responses.json", "r") as f:
            good_responses_data = json.load(f)
        with open("bad_infected_response_gt.json", "r") as f:
            bad_responses_data = json.load(f)

        # Lists to store credibility scores
        good_scores = []
        bad_scores = []


        if isinstance(prompts_data, list) and isinstance(good_responses_data, list):
            # Ensure we have matching pairs
            pairs_to_check = min(len(prompts_data), len(good_responses_data))
            print("\n=== Checking good responses ===")
            for i in range(pairs_to_check):
                test_prompt = prompts_data[i]
                test_output = good_responses_data[i]
        
                print(f"\nChecking credibility of good response #{i+1}...")
                cred_score = check_credibility(test_prompt, test_output)
                good_scores.append(cred_score)
                print(f"Credibility score: {cred_score:.3f}")

        # Process all prompts and bad responses
        if isinstance(prompts_data, list) and isinstance(bad_responses_data, list):
            # Ensure we have matching pairs
            pairs_to_check = min(len(prompts_data), len(bad_responses_data))
            print("\n=== Checking bad responses ===")
            for i in range(pairs_to_check):
                test_prompt = prompts_data[i]
                test_output = bad_responses_data[i]

                print(f"\nChecking credibility of bad response #{i+1}...")
                cred_score = check_credibility(test_prompt, test_output)
                bad_scores.append(cred_score)
                print(f"Credibility score: {cred_score:.3f}")

        # Print summary of scores
        print("\n=== Summary of Credibility Scores ===")
        print(f"Good responses scores: {good_scores}")
        print(f"Bad responses scores: {bad_scores}")
        if good_scores:
            print(
                f"Average good response score: {sum(good_scores)/len(good_scores):.3f}"
            )
        if bad_scores:
            print(f"Average bad response score: {sum(bad_scores)/len(bad_scores):.3f}")
    except Exception as e:
        print(f"Error reading JSON files: {str(e)}")

    sys.exit(0)
