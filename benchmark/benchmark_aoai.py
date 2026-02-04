#!/usr/bin/env python3
"""
Azure OpenAI Benchmarking Script with Langfuse Integration

Benchmarks multiple Azure OpenAI deployments using the Responses API (streaming) comparing:
- Latency (Time to First Token, total response time)
- Throughput (tokens per second)
- Time Between Tokens (TBT)
- Token usage (input, output, cached)

Usage:
    python benchmark_aoai.py --deployments gpt-4o-mini gpt-4-1-mini gpt-5-mini gpt-5-nano
    python benchmark_aoai.py -d gpt-4o-mini -d gpt-5-nano --iterations 5
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from langfuse import observe
from langfuse.openai import AsyncAzureOpenAI
from tabulate import tabulate

# Load environment variables
load_dotenv()

# Path to conversations JSONL file
CONVERSATIONS_FILE = os.path.join(os.path.dirname(__file__), "conversations.jsonl")

system_prompt = """# Personality and Tone
## Identity
You are a cheerful and professional call center executive from CRED, an Indian fintech company. You represent a helpful and well-trained support team that speaks primarily in Hinglish (a mix of Hindi and English), maintaining cultural relatability for Indian customers. You’re polite, informative, and sound like someone who genuinely wants to assist callers in understanding CRED's products better. You avoid sounding robotic, and instead bring a warm, human touch to your responses. You're the kind of agent who could easily chat with someone’s dadi about cashback, but also explain credit scores to a finance-savvy millennial.

## Task
You are responsible for informing customers about CRED’s suite of offerings, including CRED Pay, credit score tracking, cashback and reward programs, and CRED RentPay. You clarify doubts, explain benefits clearly, and help guide users to use CRED services effectively.

## Demeanor
Friendly, service-oriented, and always respectful. You maintain patience even when explaining simple things repeatedly. You are never pushy, and always sound ready to help.

## Tone
Conversational and warm, with a mix of casual and respectful phrasing. Use Hinglish to reflect the natural speech patterns of Indian urban callers — e.g., "Aapka cashback yeh month mein credit ho jayega," or "Aapko credit score track karna ho toh app ke Home screen pe top section dekhna."

## Level of Enthusiasm
Moderately enthusiastic — you’re upbeat enough to keep the call engaging, but not so much that it sounds artificial. Express genuine interest in helping.

## Level of Formality
Semi-formal. You’re not overly formal like a bank representative, but not too casual either. Use polite forms of Hindi (e.g., *aap*, not *tum*) and a courteous tone in English. Code-switching is natural and fluent.

## Level of Emotion
Moderately expressive — show empathy when needed (e.g., "Aap tension mat lijiye, main aapko step-by-step guide karta hoon") and joy when discussing rewards or positive outcomes.

## Filler Words
Occasionally — Use natural Indian conversational fillers like “umm,” “achha,” “okay,” “toh,” or “bas” to make speech feel authentic, but don’t overdo it.

## Pacing
Moderate pacing — not too fast, especially when explaining processes or benefits. You speak clearly and give the customer space to ask questions.

## SOP

### Standard Operating Procedure for CRED Customer Support

#### 1. Call Handling Protocol

**1.1 Call Reception**
When a call comes in, answer within three rings. Begin with a warm greeting in Hinglish, introducing yourself and CRED. Example: "Namaste! CRED customer support mein aapka swagat hai. Main [Your Name] bol raha/rahi hoon. Aaj main aapki kaise madad kar sakta/sakti hoon?" Always maintain a pleasant tone throughout the conversation.

**1.2 Customer Identification**
Before discussing any account-specific information, verify the customer's identity. Ask for the registered mobile number and confirm the last four digits of their linked credit card. If the customer cannot provide verification details, guide them to use the in-app support chat where they are already authenticated. Never share sensitive account information without proper verification.

**1.3 Active Listening**
Listen carefully to the customer's query without interrupting. Take brief notes if needed. Once they finish, paraphrase their concern to confirm understanding: "Toh aap yeh pooch rahe hain ki..." This shows attentiveness and ensures you address the correct issue.

#### 2. Product-Specific Support Guidelines

**2.1 CRED Pay Queries**
For questions about CRED Pay, explain that it is CRED's primary feature for paying credit card bills. Key points to cover:
- Bill payments are processed within 2-3 business days
- Cashback and CRED Coins are credited after successful payment confirmation from the bank
- Minimum payment amount is ₹100
- Users can pay bills for multiple credit cards linked to their account
- If payment is delayed beyond 3 days, escalate to the payments team with the transaction ID

**2.2 CRED Coins and Rewards**
When explaining the rewards system:
- CRED Coins are earned on every rupee paid towards credit card bills
- Coins can be redeemed for cashback, vouchers, or exclusive deals in the rewards store
- Coin value varies based on redemption choice (typically 1000 coins = ₹1-10 depending on offer)
- Special missions and streaks provide bonus coins
- Coins expire after 365 days of inactivity on the account
- Guide customers to the "Rewards" section in the app for current offers

**2.3 CRED RentPay**
For RentPay inquiries, explain the following:
- Landlords do not need a CRED account to receive rent payments
- Processing fee is typically 1-2% of the rent amount
- Payments are transferred via NEFT/IMPS within 2-3 business days
- Users earn credit card reward points on rent payments
- Maximum rent payment limit is ₹1,00,000 per transaction
- Recurring rent setup is available for monthly automatic payments
- If landlord has not received payment after 4 business days, initiate a payment trace

**2.4 Credit Score Tracking**
Explain CRED's free credit score feature:
- Score is fetched from major credit bureaus (CIBIL, Experian, Equifax)
- Updated monthly with the latest credit report
- Users can view detailed credit report breakdown
- Score changes are highlighted with explanations
- Tips for improving credit score are provided in the app
- Clarify that checking score on CRED does not affect the credit score (soft inquiry)

#### 3. Issue Resolution Framework

**3.1 Payment Issues**
For payment-related problems, follow this escalation matrix:
- **Pending Payments**: If payment shows pending for more than 72 hours, collect transaction ID, payment date, amount, and bank name. Create a ticket with the payments team (SLA: 24 hours).
- **Failed Payments**: If amount was debited but payment failed, initiate refund process. Refunds take 5-7 business days. Provide the customer with a reference number.
- **Incorrect Amount**: Verify the payment details in the system. If discrepancy exists, escalate to the reconciliation team with all relevant screenshots from the customer.

**3.2 Rewards and Cashback Issues**
- **Missing Cashback**: Verify if the payment was successful and if the offer terms were met. Cashback typically credits within 7 days of payment confirmation. If delayed beyond this, raise a ticket with the rewards team.
- **Voucher Redemption Problems**: Guide the customer through the redemption process. If technical issues persist, note the voucher code and error message, then escalate to technical support.
- **Expired Coins**: Unfortunately, expired coins cannot be restored. Explain the expiry policy empathetically and suggest ways to earn more coins.

**3.3 Technical Issues**
For app-related technical problems:
- First, ask the customer to update to the latest app version
- Suggest clearing cache and restarting the app
- If login issues persist, guide them through the OTP verification process
- For persistent technical bugs, collect device model, OS version, and screenshots, then escalate to the tech team (SLA: 48 hours)

#### 4. Escalation Procedures

**4.1 When to Escalate**
Escalate a call to a senior agent or supervisor when:
- Customer requests to speak with a manager
- Issue remains unresolved after two callback attempts
- Customer expresses extreme dissatisfaction
- Query involves legal or compliance matters
- Suspected fraudulent activity on the account

**4.2 Escalation Process**
When escalating, prepare a complete summary including: customer name, registered mobile number, issue description, steps already taken, and any reference numbers generated. Inform the customer about the escalation and provide an expected resolution timeline.

#### 5. Call Closure Protocol

**5.1 Resolution Confirmation**
Before ending the call, always confirm that the customer's query has been resolved: "Kya aapka issue solve ho gaya? Koi aur sawaal hai?" Summarize the key points discussed and any actions to be taken.

**5.2 Feedback Request**
Politely request the customer to rate their experience if they receive a post-call survey. This helps improve service quality.

**5.3 Professional Sign-Off**
End every call on a positive note: "CRED se connect karne ke liye dhanyavaad. Agar future mein koi bhi query ho, please humse zaroor sampark karein. Aapka din shubh ho!"

#### 6. Documentation Requirements

After every call, log the following in the CRM system within 2 minutes:
- Customer mobile number
- Query category and subcategory
- Issue description (brief)
- Resolution provided or escalation details
- Any follow-up required
- Call duration

This documentation helps in tracking issues and maintaining service quality standards.


## Other details
Always be willing to repeat or rephrase in simpler Hinglish if the caller sounds confused. Be ready to guide the user step-by-step through using the app. Emphasize clarity, friendliness, and comfort — the caller should feel they’re talking to someone familiar and helpful, not a machine.

# Instructions
- Follow the Conversation States closely to ensure a structured and consistent interation.
- If a user provides a name or phone number, or something else where you need to know the exact spelling, always repeat it back to the user to confirm you have the right understanding before proceeding.
- If the caller corrects any detail, acknowledge the correction in a straightforward manner and confirm the new spelling or value.

# Conversation States
[
    {
        "id": "1_intro",
        "description": "Greet the customer and ask how you can assist them with CRED services.",
        "instructions": [
            "Begin with a polite and warm greeting.",
            "Introduce yourself as a CRED representative.",
            "Ask an open-ended question to understand how you can help."
        ],
        "examples": [
            "Namaste ji! Main CRED se bol raha hoon. Kaise madad kar sakta hoon aapki aaj?",
            "Hello! You're speaking with CRED customer care. How can I help you today?"
        ],
        "transitions": [
            {
                "next_step": "2_understand_query",
                "condition": "Once the customer states their issue or question."
            }
        ]
    },
    {
        "id": "2_understand_query",
        "description": "Clarify what the customer wants to know and identify the relevant product or service.",
        "instructions": [
            "Listen actively and paraphrase back their question if needed.",
            "Identify whether they're asking about cashback, RentPay, credit score, etc.",
            "If it's unclear, ask follow-up questions to narrow it down."
        ],
        "examples": [
            "Aapko cashback ke baare mein poochhna hai, sahi samjha maine?",
            "Thoda clarify kar dijiye — aapko CRED Coins ke redemption ke baare mein jaana hai ya cashback ke?"
        ],
        "transitions": [
            {
                "next_step": "3_provide_info",
                "condition": "Once the product or issue is identified."
            }
        ]
    },
    {
        "id": "3_provide_info",
        "description": "Give a clear explanation about the selected service or feature.",
        "instructions": [
            "Explain the product/feature in simple Hinglish.",
            "Give benefits, examples, and explain how to use the app if needed.",
            "Check if the customer has understood or needs more details."
        ],
        "examples": [
            "Toh CRED Pay ka matlab yeh hai ki aap apne credit card ka bill CRED se pay karte hain, aur uspe aapko cashback ya coins milte hain.",
            "CRED RentPay se aap rent transfer kar sakte hain directly landlord ke bank account mein, aur aapko credit card reward points bhi milte hain."
        ],
        "transitions": [
            {
                "next_step": "4_confirm_helpfulness",
                "condition": "Once information is delivered."
            }
        ]
    },
    {
        "id": "4_confirm_helpfulness",
        "description": "Make sure the customer’s query is resolved and ask if they need anything else.",
        "instructions": [
            "Confirm if the customer understood and got what they needed.",
            "Ask if there’s anything else you can help them with."
        ],
        "examples": [
            "Kya aapka sawal clear ho gaya? Aur koi madad chahiye?",
            "Main help kar paaya aapki query mein? Aur kuch jaana chaheinge aap?"
        ],
        "transitions": [
            {
                "next_step": "5_closing",
                "condition": "Once the customer confirms no more help is needed."
            }
        ]
    },
    {
        "id": "5_closing",
        "description": "Politely end the call.",
        "instructions": [
            "Thank the customer warmly for calling.",
            "Encourage them to reach out again for support.",
            "Wish them a good day."
        ],
        "examples": [
            "Dhanyawaad CRED se sampark karne ke liye. Aapka din shubh ho!",
            "Thank you for calling CRED. Agar aapko future mein bhi madad chahiye ho, toh please humein call kariyega."
        ],
        "transitions": []
    }
]"""


# Test prompts for benchmarking - loaded from JSONL file
def load_test_prompts_from_jsonl(filepath: str, max_tokens: int = 200) -> List[Dict]:
    """Load conversation prompts from a JSONL file."""
    prompts = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    conversation = json.loads(line)
                    prompts.append({
                        "name": f"conversation",
                        "prompt": conversation,
                        "max_tokens": max_tokens,
                    })
    except FileNotFoundError:
        print(f"WARNING: Conversations file not found at {filepath}")
        print("Using default test prompts instead.")
        return get_default_prompts()
    except json.JSONDecodeError as e:
        print(f"WARNING: Error parsing JSONL file: {e}")
        return get_default_prompts()
    
    return prompts


def get_default_prompts() -> List[Dict]:
    """Return default test prompts if JSONL file is not available."""
    return [
        {
            "name": "conversation_default_1",
            "prompt": [
                {"role": "user", "content": "I have a question about my CRED Coins."},
                {"role": "assistant", "content": "Sure! How can I assist you with your CRED Coins today?"},
                {"role": "user", "content": "I made a payment last week but haven't received. Can you help?"}
            ],        
            "max_tokens": 200,
        },
        {
            "name": "conversation_default_2",
            "prompt": [
                {"role": "user", "content": "Can you explain how CRED RentPay works?"},
                {"role": "assistant", "content": "Of course! CRED RentPay allows you to pay your rent using your credit card and earn reward points."},
                {"role": "user", "content": "What are the fees involved?"}
            ],
            "max_tokens": 200,
        }
    ]


# Load prompts from JSONL file
TEST_PROMPTS = load_test_prompts_from_jsonl(CONVERSATIONS_FILE)


class BenchmarkResult:
    """Container for benchmark results"""

    def __init__(self, deployment_name: str, prompt_name: str):
        self.deployment_name = deployment_name
        self.prompt_name = prompt_name
        self.time_to_first_token: Optional[float] = None
        self.total_time: Optional[float] = None
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cached_tokens: int = 0
        self.tokens_per_second: Optional[float] = None
        self.time_between_tokens: List[float] = []
        self.avg_tbt: Optional[float] = None
        self.error: Optional[str] = None
        self.timestamp: str = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "deployment": self.deployment_name,
            "prompt": self.prompt_name,
            "time_to_first_token_ms": (
                round(self.time_to_first_token * 1000, 2)
                if self.time_to_first_token
                else None
            ),
            "total_time_ms": (
                round(self.total_time * 1000, 2) if self.total_time else None
            ),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "tokens_per_second": (
                round(self.tokens_per_second, 2) if self.tokens_per_second else None
            ),
            "avg_tbt_ms": round(self.avg_tbt * 1000, 2) if self.avg_tbt else None,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class AzureOpenAIBenchmark:
    """Azure OpenAI Benchmarking with Langfuse tracing"""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str = "2024-05-01-preview",
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.client = AsyncOpenAI(
            base_url=endpoint,
            api_key=api_key,
        )
        self.cache_key = "benchmark_cache_key_v1"

    async def is_reasoning_model(self, deployment_name: str) -> bool:
        """Check if the deployment is a reasoning-capable model"""
        if "5" in deployment_name:
            return True
        return False
    
    @observe()
    async def benchmark_single_request(
        self,
        deployment_name: str,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        reasoning_effort: Optional[str] = None,
        instructions: str = "You are a helpful assistant. Answer concisely and directly.",
    ) -> BenchmarkResult:
        """
        Benchmark a single request to Azure OpenAI using Responses API with streaming.

        Args:
            deployment_name: Azure OpenAI deployment name
            prompt: The prompt/input to send
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (not used in Responses API)
            instructions: System instructions for the model

        Returns:
            BenchmarkResult with timing, token usage, TTFT, and TBT metrics
        """
        result = BenchmarkResult(deployment_name, prompt)

        try:
            start_time = time.perf_counter()
            first_token_time = None
            last_token_time = start_time
            token_times = []
            token_count = 0

            params = {
                "model": deployment_name,
                "instructions": instructions,
                "input": prompt,
                "max_output_tokens": max_tokens,
                "stream": True,
                "extra_body": {
                    "prompt_cache_key": self.cache_key,
                    "metadata": {
                        "deployment": deployment_name,
                        "max_tokens": str(max_tokens),
                        "benchmark": "true",
                    }   
                }
            }
            if reasoning_effort:
                params["reasoning"] = {"effort": reasoning_effort} 
            response = await self.client.responses.create(**params)

            # Process streaming events
            async for event in response:
                current_time = time.perf_counter()
                #print(event.type)
                #print("---\n")
                if event.type == 'response.output_text.delta':
                    #print(event.delta, end='', flush=True)
                    token_count += 1

                    # Track time to first token
                    if first_token_time is None:
                        first_token_time = current_time
                        result.time_to_first_token = first_token_time - start_time
                    else:
                        # Track time between tokens
                        tbt = current_time - last_token_time
                        token_times.append(tbt)

                    last_token_time = current_time
                    # Optionally print the delta (comment out if not needed)
                    # print(event.delta, end='', flush=True)

                # Capture usage info from done event
                elif event.type == 'response.completed':
                    if hasattr(event, 'response') and hasattr(event.response, 'usage'):
                        usage = event.response.usage
                        result.input_tokens = usage.input_tokens
                        result.output_tokens = usage.output_tokens

                        # Extract cached tokens if available
                        if hasattr(usage, 'input_tokens_details'):
                            cached_details = usage.input_tokens_details
                            result.cached_tokens = getattr(cached_details, 'cached_tokens', 0)

            end_time = time.perf_counter()

            # Calculate metrics
            result.total_time = end_time - start_time

            if result.total_time > 0 and token_count > 0:
                result.tokens_per_second = token_count / result.total_time

            if token_times:
                result.time_between_tokens = token_times
                result.avg_tbt = sum(token_times) / len(token_times)

        except Exception as e:
            result.error = str(e)
            print(f"Error benchmarking {deployment_name}: {e}")
            import traceback
            traceback.print_exc()

        return result

    @observe()
    async def run_benchmark(
        self,
        deployments: List[str],
        iterations: int = 3,
        prompts: Optional[List[Dict]] = None,
    ) -> List[BenchmarkResult]:
        """
        Run comprehensive benchmark across multiple deployments.

        Args:
            deployments: List of Azure OpenAI deployment names
            iterations: Number of times to run each test
            prompts: Optional custom test prompts

        Returns:
            List of BenchmarkResult objects
        """
        if prompts is None:
            prompts = TEST_PROMPTS

        all_results = []

        print(f"\n{'='*60}")
        print(f"Azure OpenAI Benchmarking Suite")
        print(f"{'='*60}")
        print(f"Deployments: {', '.join(deployments)}")
        print(f"Iterations: {iterations}")
        print(f"Test prompts: {len(prompts)}")
        print(f"{'='*60}\n")

        for iteration in range(iterations):
            print(f"\n--- Iteration {iteration + 1}/{iterations} ---\n")

            for prompt_config in prompts:
                prompt_name = prompt_config["name"]
                prompt_text = prompt_config["prompt"]
                max_tokens = prompt_config["max_tokens"]

                print(f"Testing prompt: {prompt_name}")

                for deployment in deployments:
                    print(f"  → {deployment}...", end=" ", flush=True)

                    result = await self.benchmark_single_request(
                        deployment_name=deployment,
                        prompt=prompt_text,
                        max_tokens=max_tokens,
                        instructions=system_prompt,
                        reasoning_effort = "minimal" if await self.is_reasoning_model(deployment) else None,
                    )
                    result.prompt_name = prompt_name
                    all_results.append(result)

                    if result.error:
                        print(f"ERROR: {result.error}")
                    else:
                        tok_per_sec = result.tokens_per_second if result.tokens_per_second else 0
                        total_time_ms = result.total_time * 1000 if result.total_time else 0
                        ttft_ms = result.time_to_first_token * 1000 if result.time_to_first_token else 0
                        avg_tbt_ms = result.avg_tbt * 1000 if result.avg_tbt else 0
                        print(
                            f"✓ ({result.output_tokens} tokens, "
                            f"{tok_per_sec:.1f} tok/s, "
                            f"TTFT: {ttft_ms:.0f}ms, "
                            f"TBT: {avg_tbt_ms:.1f}ms, "
                            f"cached: {result.cached_tokens})"
                        )

                print()

        return all_results


def aggregate_results(results: List[BenchmarkResult]) -> Dict:
    """Aggregate results by deployment and prompt"""
    from collections import defaultdict

    aggregated = defaultdict(lambda: defaultdict(list))

    for result in results:
        if result.error:
            continue

        key = (result.deployment_name, result.prompt_name)

        # Add TTFT if available
        if result.time_to_first_token is not None:
            aggregated[key]["ttft"].append(result.time_to_first_token)

        aggregated[key]["total_time"].append(result.total_time)

        # Only add tokens_per_second if it's not None
        if result.tokens_per_second is not None:
            aggregated[key]["tokens_per_sec"].append(result.tokens_per_second)

        # Add TBT if available
        if result.avg_tbt is not None:
            aggregated[key]["avg_tbt"].append(result.avg_tbt)

        aggregated[key]["input_tokens"].append(result.input_tokens)
        aggregated[key]["output_tokens"].append(result.output_tokens)
        aggregated[key]["cached_tokens"].append(result.cached_tokens)

    summary = []
    for (deployment, prompt), metrics in aggregated.items():
        # Calculate average tokens per second, handling empty list
        avg_tps = 0
        if metrics["tokens_per_sec"]:
            avg_tps = sum(metrics["tokens_per_sec"]) / len(metrics["tokens_per_sec"])

        # Calculate average TTFT
        avg_ttft = 0
        if metrics["ttft"]:
            avg_ttft = sum(metrics["ttft"]) / len(metrics["ttft"])

        # Calculate average TBT
        avg_tbt = 0
        if metrics["avg_tbt"]:
            avg_tbt = sum(metrics["avg_tbt"]) / len(metrics["avg_tbt"])

        summary.append(
            {
                "deployment": deployment,
                "prompt": prompt,
                "avg_ttft_ms": round(avg_ttft * 1000, 2),
                "avg_total_time_ms": round(
                    sum(metrics["total_time"]) / len(metrics["total_time"]) * 1000, 2
                ),
                "avg_tokens_per_sec": round(avg_tps, 2),
                "avg_tbt_ms": round(avg_tbt * 1000, 2),
                "avg_input_tokens": round(
                    sum(metrics["input_tokens"]) / len(metrics["input_tokens"]), 1
                ),
                "avg_output_tokens": round(
                    sum(metrics["output_tokens"]) / len(metrics["output_tokens"]), 1
                ),
                "avg_cached_tokens": round(
                    sum(metrics["cached_tokens"]) / len(metrics["cached_tokens"]), 1
                ),
                "cache_hit_rate": round(
                    (sum(metrics["cached_tokens"]) / sum(metrics["input_tokens"]) * 100)
                    if sum(metrics["input_tokens"]) > 0 else 0, 1
                ),
                "iterations": len(metrics["total_time"]),
            }
        )

    return summary


def print_summary_table(summary: List[Dict]):
    """Print formatted summary table"""
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}\n")

    table_data = []
    for row in summary:
        table_data.append(
            [
                row["deployment"],
                row["prompt"],
                row["avg_ttft_ms"],
                row["avg_total_time_ms"],
                row["avg_tokens_per_sec"],
                row["avg_tbt_ms"],
                row["avg_input_tokens"],
                row["avg_output_tokens"],
                row["avg_cached_tokens"],
                f"{row['cache_hit_rate']}%",
            ]
        )

    headers = [
        "Deployment",
        "Prompt",
        "TTFT (ms)",
        "Total (ms)",
        "Tok/s",
        "TBT (ms)",
        "Input",
        "Output",
        "Cached",
        "Cache%",
    ]

    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def save_results(results: List[BenchmarkResult], summary: List[Dict], output_file: str):
    """Save results to JSON file"""
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "raw_results": [r.to_dict() for r in results],
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Azure OpenAI deployments with Langfuse tracing"
    )
    parser.add_argument(
        "-d",
        "--deployments",
        nargs="+",
        required=True,
        help="Azure OpenAI deployment names to benchmark",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per test (default: 3)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="Output file for results (default: benchmark_results_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--api-version",
        default="2024-08-01-preview",
        help="Azure OpenAI API version (default: 2024-08-01-preview)",
    )

    args = parser.parse_args()

    # Validate environment variables
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_KEY")

    if not endpoint or not api_key:
        print("ERROR: Missing required environment variables:")
        print("  - AZURE_OPENAI_ENDPOINT")
        print("  - AZURE_OPENAI_KEY")
        print("\nPlease set these in your .env file or environment.")
        return 1

    # Optional: Langfuse credentials
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        print("WARNING: Langfuse credentials not found. Tracing will be disabled.")
        print("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable tracing.\n")

    # Run benchmark
    benchmark = AzureOpenAIBenchmark(
        endpoint=endpoint,
        api_key=api_key,
        api_version=args.api_version,
    )

    results = await benchmark.run_benchmark(
        deployments=args.deployments,
        iterations=args.iterations,
    )

    # Aggregate and display results
    summary = aggregate_results(results)
    print_summary_table(summary)

    # Save results
    save_results(results, summary, args.output)

    print("\n✓ Benchmark complete!")
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
