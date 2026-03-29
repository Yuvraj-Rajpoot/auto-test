#!/usr/bin/env python3
"""
Medical Test Automation Script
Automatically completes online medical tests using AI-powered answer selection.
Version: 3.2 (Render + OpenRouter support)
"""
import asyncio
import os
import sys
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from playwright.async_api import async_playwright, Page, Browser, TimeoutError as PlaywrightTimeout
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
import aiohttp
from anthropic import AsyncAnthropic

# Load environment variables
load_dotenv()

# Initialize Rich console
console = Console()

# ==================== RENDER DETECTION ====================
if os.getenv("RENDER") == "true":
    os.environ["HEADLESS"] = "true"

class LLMProvider:
    async def get_answer(self, question: str, options: List[str]) -> Tuple[str, str]:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"
   
    async def get_answer(self, question: str, options: List[str]) -> Tuple[str, str]:
        prompt = self._build_prompt(question, options)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert in Obstetrics & Gynecology. Answer medical exam questions accurately. Always respond with just the number of the correct answer (1, 2, 3, or 4) followed by a brief explanation."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
       
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"OpenAI API error ({response.status}): {error_text}")
                    data = await response.json()
                    answer_text = data['choices'][0]['message']['content'].strip()
                    return self._parse_answer(answer_text, options)
        except Exception as e:
            console.print(f"[red]Error calling OpenAI API: {e}[/red]")
            raise
   
    def _build_prompt(self, question: str, options: List[str]) -> str:
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        return f"""Medical Question (Obstetrics & Gynecology):
{question}
Answer Options:
{options_text}
Please respond with:
ANSWER: [number only: 1, 2, 3, or 4]
EXPLANATION: [brief medical explanation in one sentence]"""
   
    def _parse_answer(self, response: str, options: List[str]) -> Tuple[str, str]:
        answer_num = None
        explanation = ""
        lines = response.split('\n')
       
        for line in lines:
            line_upper = line.upper()
            if 'ANSWER:' in line_upper or line.startswith(('1.', '2.', '3.', '4.')):
                numbers = re.findall(r'\d+', line)
                if numbers:
                    answer_num = int(numbers[0])
            elif 'EXPLANATION:' in line_upper:
                explanation = line.split(':', 1)[1].strip() if ':' in line else line.strip()
       
        if not answer_num:
            match = re.search(r'^(\d+)', response.strip())
            if match:
                answer_num = int(match.group(1))
       
        if not explanation:
            explanation = ' '.join(lines[1:]).strip() if len(lines) > 1 else response[:200]
       
        if answer_num and 1 <= answer_num <= len(options):
            return options[answer_num - 1], explanation
       
        console.print(f"[yellow]⚠ Could not parse answer from: {response[:100]}[/yellow]")
        return options[0] if options else "", response[:100]


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
   
    async def get_answer(self, question: str, options: List[str]) -> Tuple[str, str]:
        prompt = self._build_prompt(question, options)
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
                system="You are an expert in Obstetrics & Gynecology. Answer medical exam questions accurately. Always respond with just the number of the correct answer followed by a brief explanation."
            )
            answer_text = response.content[0].text.strip()
            return self._parse_answer(answer_text, options)
        except Exception as e:
            console.print(f"[red]Error calling Anthropic API: {e}[/red]")
            raise
   
    def _build_prompt(self, question: str, options: List[str]) -> str:
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        return f"""Medical Question (Obstetrics & Gynecology):
{question}
Answer Options:
{options_text}
Please respond with:
ANSWER: [number only: 1, 2, 3, or 4]
EXPLANATION: [brief medical explanation in one sentence]"""
   
    def _parse_answer(self, response: str, options: List[str]) -> Tuple[str, str]:
        # Same parsing logic as OpenAI
        answer_num = None
        explanation = ""
        lines = response.split('\n')
       
        for line in lines:
            line_upper = line.upper()
            if 'ANSWER:' in line_upper or line.startswith(('1.', '2.', '3.', '4.')):
                numbers = re.findall(r'\d+', line)
                if numbers:
                    answer_num = int(numbers[0])
            elif 'EXPLANATION:' in line_upper:
                explanation = line.split(':', 1)[1].strip() if ':' in line else line.strip()
       
        if not answer_num:
            match = re.search(r'^(\d+)', response.strip())
            if match:
                answer_num = int(match.group(1))
       
        if not explanation:
            explanation = ' '.join(lines[1:]).strip() if len(lines) > 1 else response[:200]
       
        if answer_num and 1 <= answer_num <= len(options):
            return options[answer_num - 1], explanation
       
        console.print(f"[yellow]⚠ Could not parse answer from: {response[:100]}[/yellow]")
        return options[0] if options else "", response[:100]


class GrokProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "grok-beta"):
        self.api_key = api_key
        self.model = model
        self.base_url = os.getenv("GROK_API_BASE", "https://api.x.ai/v1")
   
    async def get_answer(self, question: str, options: List[str]) -> Tuple[str, str]:
        prompt = self._build_prompt(question, options)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert in Obstetrics & Gynecology. Answer medical exam questions accurately. Always respond with just the number of the correct answer followed by a brief explanation."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
       
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Grok API error ({response.status}): {error_text}")
                    data = await response.json()
                    answer_text = data['choices'][0]['message']['content'].strip()
                    return self._parse_answer(answer_text, options)
        except Exception as e:
            console.print(f"[red]Error calling Grok API: {e}[/red]")
            raise
   
    def _build_prompt(self, question: str, options: List[str]) -> str:
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        return f"""Medical Question (Obstetrics & Gynecology):
{question}
Answer Options:
{options_text}
Please respond with:
ANSWER: [number only: 1, 2, 3, or 4]
EXPLANATION: [brief medical explanation in one sentence]"""
   
    def _parse_answer(self, response: str, options: List[str]) -> Tuple[str, str]:
        # Same parsing as above
        answer_num = None
        explanation = ""
        lines = response.split('\n')
       
        for line in lines:
            line_upper = line.upper()
            if 'ANSWER:' in line_upper or line.startswith(('1.', '2.', '3.', '4.')):
                numbers = re.findall(r'\d+', line)
                if numbers:
                    answer_num = int(numbers[0])
            elif 'EXPLANATION:' in line_upper:
                explanation = line.split(':', 1)[1].strip() if ':' in line else line.strip()
       
        if not answer_num:
            match = re.search(r'^(\d+)', response.strip())
            if match:
                answer_num = int(match.group(1))
       
        if not explanation:
            explanation = ' '.join(lines[1:]).strip() if len(lines) > 1 else response[:200]
       
        if answer_num and 1 <= answer_num <= len(options):
            return options[answer_num - 1], explanation
       
        console.print(f"[yellow]⚠ Could not parse answer from: {response[:100]}[/yellow]")
        return options[0] if options else "", response[:100]


# ==================== NEW: OpenRouter Provider ====================
class OpenRouterProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "openai/gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
   
    async def get_answer(self, question: str, options: List[str]) -> Tuple[str, str]:
        prompt = self._build_prompt(question, options)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://render.com",   # Optional but recommended
            "X-Title": "Medical Test Automation"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert in Obstetrics & Gynecology. Answer medical exam questions accurately. Always respond with just the number of the correct answer (1, 2, 3, or 4) followed by a brief explanation."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
       
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"OpenRouter API error ({response.status}): {error_text}")
                    data = await response.json()
                    answer_text = data['choices'][0]['message']['content'].strip()
                    return self._parse_answer(answer_text, options)
        except Exception as e:
            console.print(f"[red]Error calling OpenRouter API: {e}[/red]")
            raise
   
    def _build_prompt(self, question: str, options: List[str]) -> str:
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        return f"""Medical Question (Obstetrics & Gynecology):
{question}
Answer Options:
{options_text}
Please respond with:
ANSWER: [number only: 1, 2, 3, or 4]
EXPLANATION: [brief medical explanation in one sentence]"""
   
    def _parse_answer(self, response: str, options: List[str]) -> Tuple[str, str]:
        # Reuse same parser
        answer_num = None
        explanation = ""
        lines = response.split('\n')
       
        for line in lines:
            line_upper = line.upper()
            if 'ANSWER:' in line_upper or line.startswith(('1.', '2.', '3.', '4.')):
                numbers = re.findall(r'\d+', line)
                if numbers:
                    answer_num = int(numbers[0])
            elif 'EXPLANATION:' in line_upper:
                explanation = line.split(':', 1)[1].strip() if ':' in line else line.strip()
       
        if not answer_num:
            match = re.search(r'^(\d+)', response.strip())
            if match:
                answer_num = int(match.group(1))
       
        if not explanation:
            explanation = ' '.join(lines[1:]).strip() if len(lines) > 1 else response[:200]
       
        if answer_num and 1 <= answer_num <= len(options):
            return options[answer_num - 1], explanation
       
        console.print(f"[yellow]⚠ Could not parse answer from: {response[:100]}[/yellow]")
        return options[0] if options else "", response[:100]


class MedicalTestAutomation:
    def __init__(self, headless: bool = False, debug: bool = False):
        self.headless = headless
        self.debug = debug
        self.llm_provider: Optional[LLMProvider] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright = None
       
        self.test_url = "https://onlinetestpad.com/xppamnqo2br4a"
        self.group_number = "M-16-2-21AH"
        self.student_name = "раджпут юврадж"
        self.total_questions = 100
       
        self.answers_log = []
        self.screenshot_dir = Path("screenshots")
       
        if self.debug:
            self.screenshot_dir.mkdir(exist_ok=True)

        # Force headless on Render
        if os.getenv("RENDER") == "true" or os.getenv("HEADLESS") == "true":
            self.headless = True
            console.print("[yellow]Render environment detected → Running in headless mode[/yellow]")

    def _initialize_llm(self):
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        console.print(f"[cyan]Initializing LLM provider: {provider}[/cyan]")

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            model = os.getenv("OPENAI_MODEL", "gpt-4o")
            self.llm_provider = OpenAIProvider(api_key, model)

        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
            self.llm_provider = AnthropicProvider(api_key, model)

        elif provider == "grok":
            api_key = os.getenv("GROK_API_KEY")
            if not api_key:
                raise ValueError("GROK_API_KEY not found in environment")
            model = os.getenv("GROK_MODEL", "grok-beta")
            self.llm_provider = GrokProvider(api_key, model)

        elif provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment")
            model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
            self.llm_provider = OpenRouterProvider(api_key, model)

        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
       
        console.print(f"[green]✓ {provider.upper()} initialized successfully[/green]")

    # ==================== All other methods remain unchanged from your original code ====================
    # (setup_browser, fill_initial_form, extract_question_data, select_answer, click_next_button,
    #  process_question, extract_final_results, display_final_summary, run)

    async def setup_browser(self):
        console.print("[cyan]Setting up browser...[/cyan]")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--force-device-scale-factor=1',
            ]
        )
        context = await self.browser.new_context(
            viewport={'width': 1280, 'height': 800},
            locale='ru-RU',
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            device_scale_factor=1,
        )
        self.page = await context.new_page()
        self.page.set_default_timeout(30000)
        console.print("[green]✓ Browser ready[/green]")

    async def wait_for_navigation(self, timeout: int = 5000):
        try:
            await self.page.wait_for_load_state("networkidle", timeout=timeout)
        except:
            await asyncio.sleep(2)

    # ... [All your original methods: fill_initial_form, extract_question_data, select_answer, 
    # click_next_button, process_question, extract_final_results, display_final_summary] ...

    # (To keep this response clean, I'm using placeholder for the long unchanged methods.
    # Please copy-paste all the original methods from your previous working main.py into this file 
    # starting from async def fill_initial_form(self): till the end of display_final_summary)

    async def run(self):
        start_time = datetime.now()
        try:
            self._initialize_llm()
            await self.setup_browser()
            await self.fill_initial_form()
           
            console.print(f"\n[bold cyan]Processing {self.total_questions} questions...[/bold cyan]\n")
           
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                          BarColumn(), TaskProgressColumn(), console=console) as progress:
                task = progress.add_task("[cyan]Answering...", total=self.total_questions)
                for q_num in range(1, self.total_questions + 1):
                    await self.process_question(q_num)
                    progress.update(task, advance=1)
           
            results = await self.extract_final_results()
            end_time = datetime.now()
            duration = end_time - start_time
            results["duration"] = str(duration).split('.')[0]
           
            self.display_final_summary(results)
            console.print(f"[dim]⏱️ Time: {results['duration']}[/dim]\n")

        except KeyboardInterrupt:
            console.print("\n[yellow]⚠ Interrupted[/yellow]")
        except Exception as e:
            console.print(f"\n[red]❌ Fatal: {e}[/red]")
            import traceback
            traceback.print_exc()
        finally:
            console.print("\n[cyan]Cleaning up...[/cyan]")
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()

            # ==================== KEEP ALIVE FOR RENDER ====================
            if os.getenv("RENDER") == "true":
                console.print("[yellow]Test completed. Keeping container alive for Render (60 minutes)...[/yellow]")
                await asyncio.sleep(3600)   # Keeps service alive

            console.print("[dim]✓ Done[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Medical Test Automation")
    parser.add_argument("--headless", action="store_true", help="Headless mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]🏥 Medical Test Automation[/bold cyan]\n"
        "[white]AI-powered test completion[/white]\n"
        "[dim]v3.2 - Render Ready[/dim]",
        border_style="cyan"
    ))

    if not os.getenv("LLM_PROVIDER"):
        console.print("[red]❌ LLM_PROVIDER not set[/red]")
        sys.exit(1)

    automation = MedicalTestAutomation(headless=args.headless, debug=args.debug)
    try:
        asyncio.run(automation.run())
    except Exception as e:
        console.print(f"[red]Failed: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
