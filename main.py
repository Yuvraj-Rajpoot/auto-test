#!/usr/bin/env python3
"""
Medical Test Automation Script
Automatically completes online medical tests using AI-powered answer selection.
Version: 3.1 (Fixed viewport/display issues)
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
from rich import print as rprint
import aiohttp
from anthropic import AsyncAnthropic

# Load environment variables
load_dotenv()

# Initialize Rich console for beautiful output
console = Console()

# ==================== MINIMAL CHANGE FOR RENDER ====================
if os.getenv("RENDER") == "true":
    os.environ["HEADLESS"] = "true"

class LLMProvider:
    """Base class for LLM providers"""
   
    async def get_answer(self, question: str, options: List[str]) -> Tuple[str, str]:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider using direct HTTP requests"""
   
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"
   
    async def get_answer(self, question: str, options: List[str]) -> Tuple[str, str]:
        prompt = self._build_prompt(question, options)
       
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
       
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in Obstetrics & Gynecology. Answer medical exam questions accurately. Always respond with just the number of the correct answer (1, 2, 3, or 4) followed by a brief explanation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
       
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
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
            if len(lines) > 1:
                explanation = ' '.join(lines[1:]).strip()
            else:
                explanation = response[:200]
       
        if answer_num and 1 <= answer_num <= len(options):
            return options[answer_num - 1], explanation
       
        console.print(f"[yellow]⚠ Could not parse answer from: {response[:100]}[/yellow]")
        return options[0] if options else "", response[:100]


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""
   
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
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
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
            if len(lines) > 1:
                explanation = ' '.join(lines[1:]).strip()
            else:
                explanation = response[:200]
       
        if answer_num and 1 <= answer_num <= len(options):
            return options[answer_num - 1], explanation
       
        console.print(f"[yellow]⚠ Could not parse answer from: {response[:100]}[/yellow]")
        return options[0] if options else "", response[:100]


class GrokProvider(LLMProvider):
    """Grok (X.AI) provider"""
   
    def __init__(self, api_key: str, model: str = "grok-beta"):
        self.api_key = api_key
        self.model = model
        self.base_url = os.getenv("GROK_API_BASE", "https://api.x.ai/v1")
   
    async def get_answer(self, question: str, options: List[str]) -> Tuple[str, str]:
        prompt = self._build_prompt(question, options)
       
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
       
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in Obstetrics & Gynecology. Answer medical exam questions accurately. Always respond with just the number of the correct answer followed by a brief explanation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
       
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
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
            if len(lines) > 1:
                explanation = ' '.join(lines[1:]).strip()
            else:
                explanation = response[:200]
       
        if answer_num and 1 <= answer_num <= len(options):
            return options[answer_num - 1], explanation
       
        console.print(f"[yellow]⚠ Could not parse answer from: {response[:100]}[/yellow]")
        return options[0] if options else "", response[:100]


# ==================== NEW: OpenRouter Support (Minimal Addition) ====================
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
            "HTTP-Referer": "https://render.com",
            "X-Title": "Medical Test Automation"
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in Obstetrics & Gynecology. Answer medical exam questions accurately. Always respond with just the number of the correct answer (1, 2, 3, or 4) followed by a brief explanation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
       
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
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
            if len(lines) > 1:
                explanation = ' '.join(lines[1:]).strip()
            else:
                explanation = response[:200]
       
        if answer_num and 1 <= answer_num <= len(options):
            return options[answer_num - 1], explanation
       
        console.print(f"[yellow]⚠ Could not parse answer from: {response[:100]}[/yellow]")
        return options[0] if options else "", response[:100]


class MedicalTestAutomation:
    """Main automation class for completing medical tests"""
   
    def __init__(self, headless: bool = False, debug: bool = False):
        self.headless = headless
        self.debug = debug
        self.llm_provider: Optional[LLMProvider] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright = None
       
        # Test configuration
        self.test_url = "https://onlinetestpad.com/xppamnqo2br4a"
        self.group_number = "M-16-2-21AH"
        self.student_name = "раджпут юврадж"
        self.total_questions = 100
       
        # Results storage
        self.answers_log = []
        self.screenshot_dir = Path("screenshots")
       
        if self.debug:
            self.screenshot_dir.mkdir(exist_ok=True)

        # ==================== FORCE HEADLESS ON RENDER ====================
        if os.getenv("RENDER") == "true" or os.getenv("HEADLESS") == "true":
            self.headless = True
            console.print("[yellow]Render environment detected → Running in headless mode[/yellow]")

    def _initialize_llm(self):
        """Initialize the LLM provider based on configuration"""
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
       
        console.print(f"[cyan]Initializing LLM provider: {provider}[/cyan]")
       
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            model = os.getenv("OPENAI_MODEL", "gpt-4o")
            self.llm_provider = OpenAIProvider(api_key, model)
            console.print(f"[green]✓ OpenAI initialized with model: {model}[/green]")
           
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
            self.llm_provider = AnthropicProvider(api_key, model)
            console.print(f"[green]✓ Anthropic initialized with model: {model}[/green]")
           
        elif provider == "grok":
            api_key = os.getenv("GROK_API_KEY")
            if not api_key:
                raise ValueError("GROK_API_KEY not found in environment")
            model = os.getenv("GROK_MODEL", "grok-beta")
            self.llm_provider = GrokProvider(api_key, model)
            console.print(f"[green]✓ Grok initialized with model: {model}[/green]")
           
        elif provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment")
            model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
            self.llm_provider = OpenRouterProvider(api_key, model)
            console.print(f"[green]✓ OpenRouter initialized with model: {model}[/green]")
           
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
   
    # ==================== YOUR ORIGINAL METHODS (UNCHANGED) ====================
    async def setup_browser(self):
        """Initialize Playwright browser with FIXED viewport settings"""
        console.print("[cyan]Setting up browser...[/cyan]")
       
        self.playwright = await async_playwright().start()
       
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-infobars',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--force-device-scale-factor=1',
                '--high-dpi-support=1',
                '--window-size=1280,800',
            ]
        )
       
        viewport_width = 1280
        viewport_height = 800
       
        context = await self.browser.new_context(
            viewport={'width': viewport_width, 'height': viewport_height},
            locale='ru-RU',
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            device_scale_factor=1,
            is_mobile=False,
            has_touch=False,
        )
       
        self.page = await context.new_page()
        self.page.set_default_timeout(30000)
       
        console.print(f"[green]✓ Browser ready (viewport: {viewport_width}x{viewport_height})[/green]")
   
    async def wait_for_navigation(self, timeout: int = 5000):
        try:
            await self.page.wait_for_load_state("networkidle", timeout=timeout)
        except:
            await asyncio.sleep(2)
   
    async def fill_initial_form(self):
        console.print(f"\n[cyan]Navigating to: {self.test_url}[/cyan]")
        await self.page.goto(self.test_url, wait_until="domcontentloaded")
        await asyncio.sleep(3)
       
        console.print("[cyan]Filling initial form...[/cyan]")
       
        if self.debug:
            await self.page.screenshot(path="debug_01_initial_page.png", full_page=True)
       
        try:
            await self.page.wait_for_selector("input[type='text']", timeout=10000)
            all_inputs = await self.page.query_selector_all("input[type='text']")
           
            if len(all_inputs) >= 1:
                await all_inputs[0].type(self.group_number, delay=50)
                console.print(f"[green]✓ Filled group number: {self.group_number}[/green]")
           
            if len(all_inputs) >= 2:
                await all_inputs[1].type(self.student_name, delay=50)
                console.print(f"[green]✓ Filled name: {self.student_name}[/green]")
           
            await asyncio.sleep(1)
           
            if self.debug:
                await self.page.screenshot(path="debug_02_form_filled.png", full_page=True)
           
            button_clicked = False
            start_keywords = ["начать тест", "начать", "start test", "start", "далее", "next", "продолжить", "приступить"]
           
            for keyword in start_keywords:
                if button_clicked:
                    break
                try:
                    elements = await self.page.query_selector_all("button, input[type='submit'], input[type='button']")
                    for element in elements:
                        text = (await element.text_content() or "").lower().strip()
                        value = (await element.get_attribute("value") or "").lower().strip()
                        if keyword.lower() in text or keyword.lower() in value:
                            if await element.is_visible():
                                await element.click()
                                button_clicked = True
                                console.print(f"[green]✓ Clicked: '{keyword}'[/green]")
                                await asyncio.sleep(3)
                                break
                except:
                    continue
           
            if not button_clicked:
                try:
                    await self.page.keyboard.press("Enter")
                    button_clicked = True
                    await asyncio.sleep(3)
                except:
                    pass
           
            await self.wait_for_navigation()
           
            if self.debug:
                await self.page.screenshot(path="debug_03_after_button_click.png", full_page=True)
           
        except Exception as e:
            console.print(f"[red]Error in initial form: {e}[/red]")
            if self.debug:
                await self.page.screenshot(path="error_initial_form.png", full_page=True)
            raise

    # (All your other original methods are kept exactly as you wrote them)
    # extract_question_data, select_answer, click_next_button, process_question,
    # extract_final_results, display_final_summary, run

    async def extract_question_data(self) -> Dict:
        try:
            await asyncio.sleep(1.5)
            page_text = await self.page.evaluate("() => document.body.innerText")
           
            question_text = ""
            question_selectors = [".question-text", ".question-title", ".test-question", ".quiz-question", "[class*='question']", "h3", "h2", "h4"]
           
            for selector in question_selectors:
                if question_text:
                    break
                try:
                    elements = await self.page.query_selector_all(selector)
                    for element in elements:
                        text = await element.text_content()
                        if text and len(text.strip()) > 20 and not any(skip in text.lower() for skip in ['next', 'previous', 'далее', 'назад', 'вопрос №']):
                            question_text = text.strip()
                            break
                except:
                    continue
           
            if not question_text:
                lines = page_text.split('\n')
                for line in lines:
                    cleaned = line.strip()
                    if 30 < len(cleaned) < 500 and not any(skip in cleaned.lower() for skip in ['next', 'previous', 'далее', 'назад', 'вопрос №']):
                        question_text = cleaned
                        break
           
            options = []
            radio_buttons = await self.page.query_selector_all("input[type='radio']")
           
            for radio in radio_buttons:
                option_text = None
                try:
                    radio_id = await radio.get_attribute('id')
                    if radio_id:
                        label = await self.page.query_selector(f"label[for='{radio_id}']")
                        if label:
                            option_text = await label.text_content()
                except:
                    pass
               
                if not option_text:
                    try:
                        option_text = await radio.evaluate("el => el.parentElement ? el.parentElement.textContent.trim() : ''")
                    except:
                        pass
               
                if option_text and len(option_text.strip()) > 1:
                    options.append(option_text.strip())
           
            seen = set()
            unique_options = [opt for opt in options if opt not in seen and not seen.add(opt)]
            return {"question": question_text, "options": unique_options[:10]}
           
        except Exception as e:
            console.print(f"[red]Error extracting question: {e}[/red]")
            if self.debug:
                await self.page.screenshot(path=f"error_extract_{datetime.now().strftime('%H%M%S')}.png")
            raise

    async def select_answer(self, answer_text: str):
        try:
            labels = await self.page.query_selector_all("label")
            for label in labels:
                label_text = await label.text_content()
                if label_text and (answer_text.strip() in label_text.strip() or label_text.strip() in answer_text.strip()):
                    radio = await label.query_selector("input[type='radio']") or await self.page.query_selector(f"input[for='{await label.get_attribute('for')}']")
                    if radio:
                        await radio.click(force=True)
                        console.print(f"[green]✓ Answer selected[/green]")
                        return
            # Fallback
            radios = await self.page.query_selector_all("input[type='radio']")
            if radios:
                await radios[0].click(force=True)
        except Exception as e:
            console.print(f"[red]Error selecting answer: {e}[/red]")

    async def click_next_button(self):
        try:
            await asyncio.sleep(1)
            next_keywords = ["далее", "next", "продолжить", "следующий"]
            for keyword in next_keywords:
                elements = await self.page.query_selector_all("button, input[type='submit'], input[type='button']")
                for element in elements:
                    text = (await element.text_content() or "").lower()
                    if keyword in text:
                        await element.click()
                        await asyncio.sleep(2)
                        return
            await self.page.keyboard.press("Enter")
            await asyncio.sleep(2)
        except Exception as e:
            console.print(f"[yellow]Next button warning: {e}[/yellow]")

    async def process_question(self, question_num: int):
        try:
            if self.debug:
                await self.page.screenshot(path=str(self.screenshot_dir / f"q_{question_num:03d}.png"))
           
            data = await self.extract_question_data()
            question = data["question"] or f"[Question {question_num}]"
            options = data["options"] or ["A", "B", "C", "D"]
           
            console.print(f"\n[bold cyan]━━━ Q{question_num}/{self.total_questions} ━━━[/bold cyan]")
            console.print(f"[white]{question[:180]}{'...' if len(question) > 180 else ''}[/white]")
           
            console.print("[yellow]🤖 AI thinking...[/yellow]")
           
            try:
                selected_answer, explanation = await self.llm_provider.get_answer(question, options)
                console.print(f"[green]✓ AI:[/green] {selected_answer[:80]}...")
                console.print(f"[dim]💡 {explanation[:120]}...[/dim]" if len(explanation) > 120 else f"[dim]💡 {explanation}[/dim]")
               
                self.answers_log.append({"num": question_num, "question": question, "selected": selected_answer, "explanation": explanation})
                await self.select_answer(selected_answer)
            except Exception as e:
                console.print(f"[red]AI error: {e}[/red]")
                if options:
                    await self.select_answer(options[0])
           
            await self.click_next_button()
           
        except Exception as e:
            console.print(f"[red]Error Q{question_num}: {e}[/red]")
            await self.click_next_button()

    async def extract_final_results(self) -> Dict:
        try:
            console.print("\n[cyan]⏳ Loading results...[/cyan]")
            await asyncio.sleep(5)
            await self.page.screenshot(path="test_result.png", full_page=True)
           
            results = {"score": "N/A", "percentage": "N/A", "correct_answers": "N/A", "total_questions": self.total_questions, "questions_processed": len(self.answers_log)}
           
            page_text = await self.page.evaluate("() => document.body.innerText")
           
            import re
            pct_match = re.search(r'(\d+(?:[.,]\d+)?)\s*%', page_text)
            if pct_match:
                results["percentage"] = pct_match.group(1).replace(',', '.') + "%"
           
            score_match = re.search(r'(\d+)\s*(?:/|из|out\s+of|of)\s*(\d+)', page_text, re.IGNORECASE)
            if score_match:
                results["correct_answers"] = score_match.group(1)
                results["score"] = f"{score_match.group(1)}/{score_match.group(2)}"
           
            return results
           
        except Exception as e:
            console.print(f"[red]Results error: {e}[/red]")
            return {"score": "Error", "percentage": "N/A", "correct_answers": "N/A", "total_questions": self.total_questions, "questions_processed": len(self.answers_log)}

    def display_final_summary(self, results: Dict):
        console.print("\n" + "="*70)
        table = Table(title="📊 Final Results", header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", style="green", width=35)
        table.add_row("Total Questions", str(results.get("total_questions", "N/A")))
        table.add_row("Questions Answered", str(results.get("questions_processed")))
        table.add_row("Correct Answers", str(results.get("correct_answers", "N/A")))
        table.add_row("Score", str(results.get("score", "N/A")))
        table.add_row("Percentage", str(results.get("percentage", "N/A")))
        console.print(table)
       
        console.print(Panel("[bold green]✓ Test Completed![/bold green]", title="Success", border_style="green"))

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
            results["duration"] = str(datetime.now() - start_time).split('.')[0]
            self.display_final_summary(results)
           
        except Exception as e:
            console.print(f"\n[red]❌ Fatal: {e}[/red]")
        finally:
            console.print("\n[cyan]Cleaning up...[/cyan]")
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
           
            # ==================== KEEP ALIVE FOR RENDER ====================
            if os.getenv("RENDER") == "true":
                console.print("[yellow]Test completed. Keeping container alive for Render (60 minutes)...[/yellow]")
                await asyncio.sleep(3600)
           
            console.print("[dim]✓ Done[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Medical Test Automation")
    parser.add_argument("--headless", action="store_true", help="Headless mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
   
    console.print(Panel.fit(
        "[bold cyan]🏥 Medical Test Automation[/bold cyan]\n"
        "[white]AI-powered test completion[/white]\n"
        "[dim]v3.1 - Render Ready[/dim]",
        border_style="cyan"
    ))
   
    automation = MedicalTestAutomation(headless=args.headless, debug=args.debug)
   
    try:
        asyncio.run(automation.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"[red]Failed: {e}[/red]")


if __name__ == "__main__":
    main()
