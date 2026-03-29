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


class LLMProvider:
    """Base class for LLM providers"""
    
    async def get_answer(self, question: str, options: List[str]) -> Tuple[str, str]:
        """
        Get the correct answer from LLM
        Returns: (selected_option, explanation)
        """
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider using direct HTTP requests"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"
    
    async def get_answer(self, question: str, options: List[str]) -> Tuple[str, str]:
        """Get answer from OpenAI GPT using direct HTTP requests"""
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
        """Build prompt for LLM"""
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        return f"""Medical Question (Obstetrics & Gynecology):

{question}

Answer Options:
{options_text}

Please respond with:
ANSWER: [number only: 1, 2, 3, or 4]
EXPLANATION: [brief medical explanation in one sentence]"""
    
    def _parse_answer(self, response: str, options: List[str]) -> Tuple[str, str]:
        """Parse LLM response to extract answer and explanation"""
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
        
        if not answer_num:
            first_line = lines[0] if lines else response
            numbers = re.findall(r'\d+', first_line)
            if numbers:
                answer_num = int(numbers[0])
        
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
        """Get answer from Claude"""
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
        """Build prompt for LLM"""
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        return f"""Medical Question (Obstetrics & Gynecology):

{question}

Answer Options:
{options_text}

Please respond with:
ANSWER: [number only: 1, 2, 3, or 4]
EXPLANATION: [brief medical explanation in one sentence]"""
    
    def _parse_answer(self, response: str, options: List[str]) -> Tuple[str, str]:
        """Parse LLM response to extract answer and explanation"""
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
        """Get answer from Grok"""
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
        """Build prompt for LLM"""
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        return f"""Medical Question (Obstetrics & Gynecology):

{question}

Answer Options:
{options_text}

Please respond with:
ANSWER: [number only: 1, 2, 3, or 4]
EXPLANATION: [brief medical explanation in one sentence]"""
    
    def _parse_answer(self, response: str, options: List[str]) -> Tuple[str, str]:
        """Parse LLM response to extract answer and explanation"""
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
        self.context = None
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
            
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    async def setup_browser(self):
        """Initialize Playwright browser with FIXED viewport settings"""
        console.print("[cyan]Setting up browser...[/cyan]")
        
        self.playwright = await async_playwright().start()
        
        # Use a standard viewport size that fits most screens
        viewport_width = 1280
        viewport_height = 800
        
        # Create persistent user data directory for normal browser mode with extensions
        user_data_dir = Path.home() / ".playwright_chrome_profile"
        user_data_dir.mkdir(exist_ok=True)
        
        console.print(f"[dim]Using profile: {user_data_dir}[/dim]")
        
        # Launch browser with persistent context (NORMAL MODE - NOT INCOGNITO)
        # This allows extensions and maintains browser state
        self.context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            headless=self.headless,
            viewport={'width': viewport_width, 'height': viewport_height},
            locale='ru-RU',
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            device_scale_factor=1,
            is_mobile=False,
            has_touch=False,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-infobars',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--force-device-scale-factor=1',
                '--high-dpi-support=1',
                '--window-size=1280,800',
                '--enable-extensions',
            ],
            ignore_default_args=['--disable-extensions'],
        )
        
        # Get the first page or create a new one
        if self.context.pages:
            self.page = self.context.pages[0]
        else:
            self.page = await self.context.new_page()
        
        # Set default timeout
        self.page.set_default_timeout(30000)
        
        console.print(f"[green]✓ Browser ready (NORMAL MODE with extensions, viewport: {viewport_width}x{viewport_height})[/green]")
        
        # ═══════════════════════════════════════════════════════
        # WAIT FOR USER TO CONNECT VPN
        # ═══════════════════════════════════════════════════════
        console.print("\n" + "="*50)
        console.print("[bold yellow]⚠️  VPN CONNECTION REQUIRED[/bold yellow]")
        console.print("="*50)
        console.print("[cyan]1. Connect your VPN extension now[/cyan]")
        console.print("[cyan]2. Wait for VPN to fully connect[/cyan]")
        console.print("[cyan]3. Press ENTER to continue...[/cyan]")
        console.print("="*50 + "\n")
        
        # Wait for user input
        await asyncio.get_event_loop().run_in_executor(None, input)
        
        console.print("[green]✓ Continuing with test...[/green]\n")
    
    async def wait_for_navigation(self, timeout: int = 5000):
        """Wait for page navigation with timeout"""
        try:
            await self.page.wait_for_load_state("networkidle", timeout=timeout)
        except:
            await asyncio.sleep(2)
    
    async def fill_initial_form(self):
        """Fill the initial form with group number and name"""
        console.print(f"\n[cyan]Navigating to: {self.test_url}[/cyan]")
        
        await self.page.goto(self.test_url, wait_until="domcontentloaded")
        await asyncio.sleep(3)
        
        console.print("[cyan]Filling initial form...[/cyan]")
        
        # Take initial screenshot for debugging
        if self.debug:
            await self.page.screenshot(path="debug_01_initial_page.png", full_page=True)
            console.print("[dim]📸 Saved: debug_01_initial_page.png[/dim]")
        
        try:
            # Wait for form fields
            await self.page.wait_for_selector("input[type='text']", timeout=10000)
            all_inputs = await self.page.query_selector_all("input[type='text']")
            
            console.print(f"[dim]Found {len(all_inputs)} text input fields[/dim]")
            
            if len(all_inputs) >= 1:
                await all_inputs[0].click()
                await all_inputs[0].fill("")
                await asyncio.sleep(0.3)
                await all_inputs[0].type(self.group_number, delay=50)
                console.print(f"[green]✓ Filled group number: {self.group_number}[/green]")
            
            if len(all_inputs) >= 2:
                await all_inputs[1].click()
                await all_inputs[1].fill("")
                await asyncio.sleep(0.3)
                await all_inputs[1].type(self.student_name, delay=50)
                console.print(f"[green]✓ Filled name: {self.student_name}[/green]")
            
            await asyncio.sleep(1)
            
            if self.debug:
                await self.page.screenshot(path="debug_02_form_filled.png", full_page=True)
                console.print("[dim]📸 Saved: debug_02_form_filled.png[/dim]")
            
            # ═══════════════════════════════════════════════════════
            # FIND AND CLICK START BUTTON
            # ═══════════════════════════════════════════════════════
            
            console.print("[cyan]Looking for start button...[/cyan]")
            button_clicked = False
            
            # Strategy 1: Common button text patterns
            start_keywords = [
                "начать тест", "начать", "start test", "start", 
                "далее", "next", "продолжить", "приступить", 
                "go", "begin", "submit", "отправить"
            ]
            
            for keyword in start_keywords:
                if button_clicked:
                    break
                try:
                    elements = await self.page.query_selector_all(
                        "button, input[type='submit'], input[type='button'], a.btn, .button"
                    )
                    for element in elements:
                        try:
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
                except:
                    continue
            
            # Strategy 2: Submit buttons
            if not button_clicked:
                try:
                    submit_buttons = await self.page.query_selector_all("input[type='submit'], button[type='submit']")
                    for btn in submit_buttons:
                        if await btn.is_visible():
                            await btn.click()
                            button_clicked = True
                            console.print("[green]✓ Clicked submit button[/green]")
                            await asyncio.sleep(3)
                            break
                except:
                    pass
            
            # Strategy 3: Primary buttons by class
            if not button_clicked:
                primary_selectors = [
                    ".btn-primary", ".btn-success", ".btn-action",
                    ".primary-button", ".submit-button", ".start-button",
                    "button.btn", ".next-btn", "[class*='primary']"
                ]
                
                for selector in primary_selectors:
                    if button_clicked:
                        break
                    try:
                        buttons = await self.page.query_selector_all(selector)
                        for btn in buttons:
                            if await btn.is_visible():
                                await btn.click()
                                button_clicked = True
                                console.print(f"[green]✓ Clicked: {selector}[/green]")
                                await asyncio.sleep(3)
                                break
                    except:
                        continue
            
            # Strategy 4: Any visible button
            if not button_clicked:
                all_buttons = await self.page.query_selector_all(
                    "button, input[type='submit'], input[type='button']"
                )
                
                for idx, button in enumerate(all_buttons):
                    try:
                        if not await button.is_visible():
                            continue
                        
                        text = (await button.text_content() or "").strip()
                        value = (await button.get_attribute("value") or "").strip()
                        
                        # Skip cancel/close buttons
                        skip_patterns = ['close', 'cancel', 'back', 'назад', 'отмена', 'закрыть']
                        if any(pattern in text.lower() or pattern in value.lower() for pattern in skip_patterns):
                            continue
                        
                        await button.click()
                        button_clicked = True
                        console.print(f"[yellow]⚠ Clicked button: '{text or value}'[/yellow]")
                        await asyncio.sleep(3)
                        break
                    except:
                        continue
            
            # Strategy 5: Press Enter
            if not button_clicked:
                try:
                    await self.page.keyboard.press("Enter")
                    button_clicked = True
                    console.print("[green]✓ Pressed Enter[/green]")
                    await asyncio.sleep(3)
                except:
                    pass
            
            if not button_clicked:
                console.print("[red]✗ Could not find start button![/red]")
            
            # Wait for page to load
            await self.wait_for_navigation()
            
            if self.debug:
                await self.page.screenshot(path="debug_03_after_button_click.png", full_page=True)
            
            # Verify test started
            await asyncio.sleep(2)
            radio_buttons = await self.page.query_selector_all("input[type='radio']")
            
            if len(radio_buttons) > 0:
                console.print(f"[green]✓✓✓ Test started! Found {len(radio_buttons)} options[/green]")
            else:
                console.print("[yellow]⚠ Waiting for test to load...[/yellow]")
                await asyncio.sleep(3)
                
                radio_buttons = await self.page.query_selector_all("input[type='radio']")
                if len(radio_buttons) > 0:
                    console.print(f"[green]✓ Test loaded! Found {len(radio_buttons)} options[/green]")
                else:
                    console.print("[red]⚠ No radio buttons found. Check debug screenshots.[/red]")
            
        except Exception as e:
            console.print(f"[red]Error in initial form: {e}[/red]")
            if self.debug:
                await self.page.screenshot(path="error_initial_form.png", full_page=True)
            raise
    
    async def extract_question_data(self) -> Dict:
        """Extract question text and answer options from current page"""
        try:
            await asyncio.sleep(1.5)
            
            page_text = await self.page.evaluate("() => document.body.innerText")
            
            # ═══════════════════════════════════════════════════════
            # EXTRACT QUESTION TEXT
            # ═══════════════════════════════════════════════════════
            
            question_text = ""
            
            question_selectors = [
                ".question-text", ".question-title", ".test-question",
                ".quiz-question", "[class*='question']", ".q-text",
                "h3", "h2", "h4", ".task"
            ]
            
            for selector in question_selectors:
                if question_text:
                    break
                try:
                    elements = await self.page.query_selector_all(selector)
                    for element in elements:
                        text = await element.text_content()
                        if text and len(text.strip()) > 20:
                            if not any(skip in text.lower() for skip in ['next', 'previous', 'далее', 'назад', 'вопрос №']):
                                question_text = text.strip()
                                break
                except:
                    continue
            
            # Fallback: Parse from page text
            if not question_text:
                lines = page_text.split('\n')
                for line in lines:
                    cleaned = line.strip()
                    if len(cleaned) > 30 and len(cleaned) < 500:
                        if not any(skip in cleaned.lower() for skip in ['next', 'previous', 'далее', 'назад', 'вопрос №']):
                            question_text = cleaned
                            break
            
            # ═══════════════════════════════════════════════════════
            # EXTRACT ANSWER OPTIONS
            # ═══════════════════════════════════════════════════════
            
            options = []
            radio_buttons = await self.page.query_selector_all("input[type='radio']")
            
            for radio in radio_buttons:
                option_text = None
                
                # Method 1: Label by 'for' attribute
                try:
                    radio_id = await radio.get_attribute('id')
                    if radio_id:
                        label = await self.page.query_selector(f"label[for='{radio_id}']")
                        if label:
                            option_text = await label.text_content()
                except:
                    pass
                
                # Method 2: Parent label
                if not option_text:
                    try:
                        parent_label = await radio.evaluate_handle("el => el.closest('label')")
                        if parent_label:
                            option_text = await parent_label.evaluate("el => el.textContent")
                    except:
                        pass
                
                # Method 3: Next sibling text
                if not option_text:
                    try:
                        option_text = await radio.evaluate("""
                            el => {
                                let next = el.nextSibling;
                                while (next) {
                                    if (next.nodeType === 3 && next.textContent.trim()) {
                                        return next.textContent.trim();
                                    }
                                    if (next.nodeType === 1 && next.textContent.trim()) {
                                        return next.textContent.trim();
                                    }
                                    next = next.nextSibling;
                                }
                                if (el.parentElement) {
                                    return el.parentElement.textContent.trim();
                                }
                                return '';
                            }
                        """)
                    except:
                        pass
                
                if option_text and option_text.strip() and len(option_text.strip()) > 1:
                    options.append(option_text.strip())
            
            # Remove duplicates
            seen = set()
            unique_options = []
            for opt in options:
                cleaned_opt = opt.strip()
                if len(cleaned_opt) >= 2 and cleaned_opt not in seen:
                    seen.add(cleaned_opt)
                    unique_options.append(cleaned_opt)
            
            options = unique_options[:10]
            
            return {
                "question": question_text,
                "options": options
            }
            
        except Exception as e:
            console.print(f"[red]Error extracting question: {e}[/red]")
            if self.debug:
                await self.page.screenshot(path=f"error_extract_{datetime.now().strftime('%H%M%S')}.png")
            raise
    
    async def select_answer(self, answer_text: str):
        """Select the radio button corresponding to the answer"""
        try:
            labels = await self.page.query_selector_all("label")
            
            for label in labels:
                try:
                    label_text = await label.text_content()
                    if not label_text:
                        continue
                    
                    # Fuzzy match
                    if answer_text.strip() in label_text.strip() or label_text.strip() in answer_text.strip():
                        radio = await label.query_selector("input[type='radio']")
                        
                        if not radio:
                            label_for = await label.get_attribute('for')
                            if label_for:
                                radio = await self.page.query_selector(f"input[id='{label_for}']")
                        
                        if radio:
                            is_checked = await radio.is_checked()
                            if not is_checked:
                                try:
                                    await radio.click(force=True)
                                    await asyncio.sleep(0.5)
                                    console.print(f"[green]✓ Answer selected[/green]")
                                    return
                                except:
                                    await label.click(force=True)
                                    await asyncio.sleep(0.5)
                                    console.print(f"[green]✓ Answer selected (via label)[/green]")
                                    return
                            else:
                                return
                except:
                    continue
            
            # Fallback: click first radio
            console.print(f"[yellow]⚠ Using fallback selection[/yellow]")
            radio_buttons = await self.page.query_selector_all("input[type='radio']")
            if radio_buttons:
                try:
                    await radio_buttons[0].click(force=True)
                    await asyncio.sleep(0.5)
                except:
                    pass
            
        except Exception as e:
            console.print(f"[red]Error selecting answer: {e}[/red]")
    
    async def click_next_button(self):
        """Click the 'Next' button to proceed"""
        try:
            await asyncio.sleep(1)
            
            next_keywords = [
                "далее", "next", "продолжить", "следующий",
                "continue", "proceed", "go on", ">"
            ]
            
            button_clicked = False
            
            for keyword in next_keywords:
                if button_clicked:
                    break
                try:
                    elements = await self.page.query_selector_all(
                        "button, input[type='submit'], input[type='button']"
                    )
                    for element in elements:
                        try:
                            if not await element.is_visible():
                                continue
                            text = (await element.text_content() or "").lower().strip()
                            value = (await element.get_attribute("value") or "").lower().strip()
                            
                            if keyword.lower() in text or keyword.lower() in value:
                                await element.click()
                                button_clicked = True
                                await asyncio.sleep(2)
                                return
                        except:
                            continue
                except:
                    continue
            
            # Fallback: submit button
            if not button_clicked:
                try:
                    submit_btns = await self.page.query_selector_all("button[type='submit'], input[type='submit']")
                    for btn in submit_btns:
                        if await btn.is_visible():
                            await btn.click()
                            await asyncio.sleep(2)
                            return
                except:
                    pass
            
            # Fallback: Enter key
            if not button_clicked:
                try:
                    await self.page.keyboard.press("Enter")
                    await asyncio.sleep(2)
                except:
                    pass
            
        except Exception as e:
            console.print(f"[yellow]Next button warning: {e}[/yellow]")
    
    async def process_question(self, question_num: int):
        """Process a single question"""
        try:
            if self.debug:
                screenshot_path = self.screenshot_dir / f"q_{question_num:03d}.png"
                await self.page.screenshot(path=str(screenshot_path))
            
            data = await self.extract_question_data()
            question = data["question"]
            options = data["options"]
            
            if not question:
                question = f"[Question {question_num}]"
            
            if not options or len(options) < 2:
                if not options:
                    options = ["A", "B", "C", "D"]
            
            # Display
            console.print(f"\n[bold cyan]━━━ Q{question_num}/{self.total_questions} ━━━[/bold cyan]")
            display_q = question[:180] + "..." if len(question) > 180 else question
            console.print(f"[white]{display_q}[/white]")
            console.print(f"[dim]Options: {len(options)}[/dim]")
            
            # Get AI answer
            console.print("[yellow]🤖 AI thinking...[/yellow]")
            
            try:
                selected_answer, explanation = await self.llm_provider.get_answer(question, options)
                
                display_ans = selected_answer[:80] + "..." if len(selected_answer) > 80 else selected_answer
                console.print(f"[green]✓ AI:[/green] {display_ans}")
                console.print(f"[dim]💡 {explanation[:120]}...[/dim]" if len(explanation) > 120 else f"[dim]💡 {explanation}[/dim]")
                
                self.answers_log.append({
                    "num": question_num,
                    "question": question,
                    "selected": selected_answer,
                    "explanation": explanation
                })
                
                await self.select_answer(selected_answer)
                
            except Exception as e:
                console.print(f"[red]AI error: {e}[/red]")
                if options:
                    await self.select_answer(options[0])
                    self.answers_log.append({
                        "num": question_num,
                        "question": question,
                        "selected": options[0],
                        "explanation": f"Fallback: {str(e)[:50]}"
                    })
            
            await self.click_next_button()
            
        except Exception as e:
            console.print(f"[red]Error Q{question_num}: {e}[/red]")
            try:
                await self.click_next_button()
            except:
                pass
    
    async def extract_final_results(self) -> Dict:
        """Extract final test results"""
        try:
            console.print("\n[cyan]⏳ Loading results...[/cyan]")
            await asyncio.sleep(5)
            
            await self.page.screenshot(path="test_result.png", full_page=True)
            console.print("[green]✓ Screenshot: test_result.png[/green]")
            
            results = {
                "score": "N/A",
                "percentage": "N/A",
                "correct_answers": "N/A",
                "total_questions": self.total_questions,
                "questions_processed": len(self.answers_log)
            }
            
            page_text = await self.page.evaluate("() => document.body.innerText")
            
            if self.debug:
                with open("debug_results.txt", "w", encoding="utf-8") as f:
                    f.write(page_text)
            
            # Extract patterns
            pct_match = re.search(r'(\d+(?:[.,]\d+)?)\s*%', page_text)
            if pct_match:
                results["percentage"] = pct_match.group(1).replace(',', '.') + "%"
            
            score_match = re.search(r'(\d+)\s*(?:/|из|out\s+of|of)\s*(\d+)', page_text, re.IGNORECASE)
            if score_match:
                results["correct_answers"] = score_match.group(1)
                results["total_questions"] = score_match.group(2)
                results["score"] = f"{score_match.group(1)}/{score_match.group(2)}"
            
            correct_match = re.search(r'(?:correct|правильно|верно)[:\s]+(\d+)', page_text, re.IGNORECASE)
            if correct_match:
                results["correct_answers"] = correct_match.group(1)
            
            return results
            
        except Exception as e:
            console.print(f"[red]Results error: {e}[/red]")
            try:
                await self.page.screenshot(path="test_result.png", full_page=True)
            except:
                pass
            return {
                "score": "Error",
                "percentage": "N/A",
                "correct_answers": "N/A",
                "total_questions": self.total_questions,
                "questions_processed": len(self.answers_log)
            }
    
    def display_final_summary(self, results: Dict):
        """Display final summary"""
        console.print("\n" + "="*70)
        
        table = Table(
            title="📊 Final Results",
            show_header=True,
            header_style="bold magenta",
            border_style="cyan"
        )
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", style="green", width=35)
        
        table.add_row("Total Questions", str(results.get("total_questions", "N/A")))
        table.add_row("Questions Answered", str(results.get("questions_processed", len(self.answers_log))))
        table.add_row("Correct Answers", str(results.get("correct_answers", "N/A")))
        table.add_row("Score", str(results.get("score", "N/A")))
        table.add_row("Percentage", str(results.get("percentage", "N/A")))
        table.add_row("Screenshot", "test_result.png")
        
        console.print(table)
        
        panel = Panel(
            "[bold green]✓ Test Completed![/bold green]\n\n"
            f"• Questions answered: {len(self.answers_log)}\n"
            f"• Results saved: [cyan]test_result.png[/cyan]",
            title="✨ Success",
            border_style="green",
            padding=(1, 2)
        )
        console.print(panel)
        console.print("="*70 + "\n")
    
    async def run(self):
        """Main execution flow"""
        start_time = datetime.now()
        
        try:
            self._initialize_llm()
            await self.setup_browser()
            await self.fill_initial_form()
            
            console.print(f"\n[bold cyan]Processing {self.total_questions} questions...[/bold cyan]\n")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task(
                    "[cyan]Answering...",
                    total=self.total_questions
                )
                
                for q_num in range(1, self.total_questions + 1):
                    await self.process_question(q_num)
                    progress.update(task, advance=1)
            
            results = await self.extract_final_results()
            
            end_time = datetime.now()
            duration = end_time - start_time
            results["duration"] = str(duration).split('.')[0]
            
            self.display_final_summary(results)
            console.print(f"[dim]⏱️ Time: {results['duration']}[/dim]\n")
            
            if not self.headless:
                console.print("[yellow]Browser open for 15 seconds...[/yellow]")
                await asyncio.sleep(15)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠ Interrupted[/yellow]")
            if len(self.answers_log) > 0:
                console.print(f"[cyan]Processed {len(self.answers_log)} questions[/cyan]")
            
        except Exception as e:
            console.print(f"\n[red]❌ Fatal: {e}[/red]")
            import traceback
            traceback.print_exc()
            
            try:
                if self.page:
                    await self.page.screenshot(path="error_fatal.png", full_page=True)
            except:
                pass
                
        finally:
            console.print("\n[cyan]Cleaning up...[/cyan]")
            if self.context:
                await self.context.close()
            if self.playwright:
                await self.playwright.stop()
            console.print("[dim]✓ Done[/dim]")


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(
        description="Medical Test Automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Visible browser
  python main.py --headless         # Background mode
  python main.py --debug            # Save screenshots
        """
    )
    
    parser.add_argument("--headless", action="store_true", help="Headless mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    # Banner
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]🏥 Medical Test Automation[/bold cyan]\n"
        "[white]AI-powered test completion[/white]\n"
        "[dim]v3.1[/dim]",
        border_style="cyan"
    ))
    console.print("\n")
    
    # Validate
    if not os.getenv("LLM_PROVIDER"):
        console.print("[red]❌ LLM_PROVIDER not set in .env[/red]")
        sys.exit(1)
    
    provider = os.getenv("LLM_PROVIDER", "").lower()
    api_keys = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "grok": "GROK_API_KEY"}
    
    if provider in api_keys and not os.getenv(api_keys[provider]):
        console.print(f"[red]❌ {api_keys[provider]} not set[/red]")
        sys.exit(1)
    
    # Run
    automation = MedicalTestAutomation(headless=args.headless, debug=args.debug)
    
    try:
        asyncio.run(automation.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Failed: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()