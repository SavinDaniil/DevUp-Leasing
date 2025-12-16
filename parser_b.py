"""
Leasing asset market analyzer (CLI).

Key improvements for Sber use‑case:
- Dedicated Avito list-page parser (HTML only, no LLM).
- Reusable Selenium driver with configurable scroll depth.
- Safe JSON parsing of LLM output.
- JSON-LD extraction as a structured data source.
- Market price analysis with outlier filtering (IQR).
"""

import io
import json
import os
import re
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional, Any
from urllib.parse import urljoin, urlparse

import requests
import urllib3
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from dotenv import load_dotenv
import logging

load_dotenv()

# Настройте logger (можно добавить после load_dotenv())
logger = logging.getLogger(__name__)

# Ensure stdout handles UTF-8 on Windows to avoid mojibake in console.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GIGACHAT_AUTH_DATA = os.getenv("GIGACHAT_AUTH_DATA")


# =============================
# Data structures
# =============================
@dataclass
class LeasingOffer:
    title: str
    url: str
    source: str
    model: str = ""
    price: Optional[int] = None  # numeric price
    price_str: Optional[str] = None  # formatted price string
    monthly_payment: Optional[int] = None
    monthly_payment_str: Optional[str] = None
    price_on_request: bool = False
    year: Optional[int] = None
    power: Optional[str] = None
    mileage: Optional[str] = None
    vendor: Optional[str] = None
    condition: Optional[str] = None
    location: Optional[str] = None
    specs: dict = field(default_factory=dict)
    category: Optional[str] = None
    currency: Optional[str] = None
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    analogs: list[str] = field(default_factory=list)
    analogs_suggested: list[str] = field(default_factory=list)

    def has_data(self) -> bool:
        return any(
            [
                self.price is not None,
                self.monthly_payment is not None,
                self.price_on_request,
            ]
        )


# =============================
# Utility helpers
# =============================
def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def digits_to_int(text: str) -> Optional[int]:
    digits = re.sub(r"[^\d]", "", text or "")
    if not digits:
        return None
    # Avoid parsing phone numbers or long concatenations as integers
    if len(digits) > 16:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def format_price(value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    return f"{value:,}".replace(",", " ") + " ₽"


def ensure_list_str(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def extract_price_candidate(text: str) -> Optional[int]:
    """
    Smarter extraction: looks for numbers followed by currency markers (₽, руб, rub, $, €).
    If no currency, looks for large numbers (>10000) that aren't years (19xx, 20xx).
    Filters out unlikely values (too small or too huge).
    """
    if not text:
        return None
    
    # Try currency patterns first
    cur_pattern = r"(\d[\d\s]*)\s*(₽|руб|rub|\$|€)"
    matches = re.findall(cur_pattern, text, flags=re.IGNORECASE)
    for m in matches:
        val = digits_to_int(m[0])
        # Reasonable bounds: > 1000 RUB, < 100 billion
        if val and 1000 < val < 100_000_000_000:
            return val
            
    # Fallback: look for generic big numbers, avoiding years
    # 4-digit numbers starting with 19 or 20 are usually years
    nums = re.findall(r"\b\d[\d\s]*\b", text)
    for n in nums:
        val = digits_to_int(n)
        if not val: 
            continue
        # Avoid likely years
        if 1900 <= val <= 2030:
            continue
        # Assume valid price is likely > 10000 (context dependent, but safe for machinery/cars)
        # and < 100 billion
        if 10000 < val < 100_000_000_000:
            return val
    return None


def normalize_url(url: str, base: str = "https://www.avito.ru") -> str:
    if not url:
        return url
    if url.startswith("http"):
        return url
    return urljoin(base, url)


def is_relevant_avito_title(title: str, model_name: str) -> bool:
    """Check if all model keywords are present in title."""
    if not model_name:
        return True
    title_l = title.lower()
    keywords = [w for w in re.split(r"\s+", model_name.lower()) if w]
    return all(k in title_l for k in keywords)


def safe_json_loads(content: str) -> Optional[Any]:
    """
    Robust JSON loader for LLM output.
    - strips code fences
    - finds first {...} or [...] block
    - returns None on failure
    """
    if not content:
        return None
    cleaned = content.replace("```json", "").replace("```", "").strip()
    
    # Try direct parse first
    try:
        return json.loads(cleaned)
    except:
        pass

    # Find potential JSON start/end
    start_brace = cleaned.find("{")
    start_bracket = cleaned.find("[")
    
    # Determine which structure comes first
    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        start = start_brace
        end = cleaned.rfind("}")
    elif start_bracket != -1 and (start_brace == -1 or start_bracket < start_brace):
        start = start_bracket
        end = cleaned.rfind("]")
    else:
        return None

    if start == -1 or end == -1 or end <= start:
        return None
        
    candidate = cleaned[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def is_specialized_equipment(query: str) -> bool:
    """
    Checks if the query describes specialized/industrial equipment
    that requires specific search strategies.
    """
    specialized_keywords = [
        "буровая", "буровое", "буровой", "бурильная",
        "экскаватор", "крановое", "подъемное",
        "промышленное", "производственное", "технологическое",
        "компрессор", "генератор", "насосное",
        "сварочное", "обрабатывающее", "станок"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in specialized_keywords)


# =============================
# Avito parsing
# =============================
def _extract_year_from_text(text: str) -> Optional[int]:
    m = re.search(r"(20[0-4]\d|19\d{2})", text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _extract_power(text: str) -> Optional[str]:
    m = re.search(r"(\d{2,4})\s*(л\.?с\.?|hp)", text, flags=re.IGNORECASE)
    return m.group(1) if m else None


def _extract_mileage(text: str) -> Optional[str]:
    m = re.search(r"(\d[\d\s]{2,6})\s*(км|km)", text, flags=re.IGNORECASE)
    return normalize_whitespace(m.group(0)) if m else None


def extract_offers_from_ld_json(html: str) -> list[dict]:
    offers = []
    soup = BeautifulSoup(html or "", "html.parser")
    scripts = soup.find_all("script", {"type": "application/ld+json"})
    for script in scripts:
        data = safe_json_loads(script.get_text(strip=True))
        if not data:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("@type", "")
            if isinstance(item_type, list):
                item_type = ",".join(item_type)
            if "Offer" in item_type or "Product" in item_type:
                offers.append(item)
            # Some Avito pages embed LD inside "itemListElement"
            if "itemListElement" in item and isinstance(item["itemListElement"], list):
                for sub in item["itemListElement"]:
                    if isinstance(sub, dict) and "item" in sub and isinstance(sub["item"], dict):
                        offers.append(sub["item"])
    return offers


def parse_avito_list_page(html: str, model_name: str) -> list[LeasingOffer]:
    soup = BeautifulSoup(html or "", "html.parser")
    cards = soup.select('[data-marker="item"]')
    if not cards:
        cards = soup.select("div.iva-item-root")

    offers: list[LeasingOffer] = []
    for card in cards:
        title_tag = card.select_one('[data-marker="item-title"]') or card.select_one("a")
        if not title_tag:
            continue
        title = normalize_whitespace(title_tag.get_text(" ", strip=True))
        if not is_relevant_avito_title(title, model_name):
            continue

        href = title_tag.get("href") or ""
        url = normalize_url(href)
        price_tag = card.select_one('[data-marker="item-price"]') or card.select_one("meta[itemprop='price']")
        price_val = None
        price_display = None
        if price_tag:
            price_text = price_tag.get("content") or price_tag.get_text(" ", strip=True)
            price_val = digits_to_int(price_text)
            price_display = format_price(price_val) if price_val else None

        location_tag = card.select_one('[data-marker="item-location"]')
        location = normalize_whitespace(location_tag.get_text(" ", strip=True)) if location_tag else None

        subtitle = card.get_text(" ", strip=True)
        year = _extract_year_from_text(subtitle)
        power = _extract_power(subtitle)
        mileage = _extract_mileage(subtitle)

        offers.append(
            LeasingOffer(
                title=title,
                url=url,
                source="avito.ru",
                model=model_name,
                price=price_val,
                price_str=price_display,
                location=location,
                year=year,
                power=power,
                mileage=mileage,
            )
        )

    # LD-JSON as supplemental source
    for item in extract_offers_from_ld_json(html):
        name = normalize_whitespace(item.get("name", "") or item.get("title", ""))
        href = item.get("url") or item.get("mainEntityOfPage") or ""
        price_val = digits_to_int(str(item.get("price", "")))
        if not name and not href:
            continue
        if not is_relevant_avito_title(name, model_name):
            continue
        offers.append(
            LeasingOffer(
                title=name or "Offer",
                url=normalize_url(href),
                source="avito.ru",
                model=model_name,
                price=price_val,
                price_str=format_price(price_val),
                location=normalize_whitespace(item.get("address", "")) or None,
                year=_extract_year_from_text(name),
            )
        )

    return offers


# =============================
# Selenium handling
# =============================
class SeleniumFetcher:
    def __init__(self):
        self.driver: Optional[webdriver.Chrome] = None
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless=new")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--window-size=1920,1080")
        self.chrome_options.add_argument("--log-level=3")
        self.chrome_options.add_argument("--disable-logging")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        # Убираем предупреждение о WebGL fallback
        self.chrome_options.add_argument("--disable-webgl")
        self.chrome_options.add_argument("--disable-software-rasterizer")
        # Альтернативный вариант (как предлагает само сообщение):
        # self.chrome_options.add_argument("--enable-unsafe-swiftshader")
        self.chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
        # Подавляем вывод DevTools и других предупреждений
        self.chrome_options.add_experimental_option('excludeSwitches', ['enable-logging', 'enable-automation'])
        self.chrome_options.add_experimental_option('useAutomationExtension', False)
        self.chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        )
        # Устанавливаем таймауты
        self.page_load_timeout = 30
        self.implicit_wait = 10

    def _is_driver_alive(self) -> bool:
        """Проверяет, жив ли драйвер и может ли выполнять команды."""
        if not self.driver:
            return False
        try:
            # Пробуем выполнить простую команду
            self.driver.current_url
            return True
        except Exception:
            return False

    def _get_driver(self) -> webdriver.Chrome:
        """Получает рабочий драйвер, пересоздавая при необходимости."""
        if self._is_driver_alive():
            return self.driver
        
        # Если драйвер мертв, закрываем его и создаем новый
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
        
        # Создаем новый драйвер
        self.driver = webdriver.Chrome(options=self.chrome_options)
        self.driver.set_page_load_timeout(self.page_load_timeout)
        self.driver.implicitly_wait(self.implicit_wait)
        return self.driver

    def close(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            finally:
                self.driver = None

    def fetch_page(self, url: str, scroll_times: int = 2, wait: float = 1.5, max_retries: int = 2) -> Optional[str]:
        """
        Загружает страницу с retry логикой.
        
        Args:
            url: URL для загрузки
            scroll_times: Количество прокруток
            wait: Время ожидания между прокрутками
            max_retries: Максимальное количество попыток при ошибке соединения
        """
        for attempt in range(max_retries + 1):
            driver = None
            try:
                driver = self._get_driver()
                driver.get(url)
                last_height = driver.execute_script("return document.body.scrollHeight")
                for _ in range(max(0, scroll_times)):
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(wait)
                    new_height = driver.execute_script("return document.body.scrollHeight")
                    if new_height == last_height:
                        break
                    last_height = new_height
                return driver.page_source
            except Exception as e:
                error_msg = str(e)
                # Проверяем, является ли это ошибкой соединения
                is_connection_error = any(keyword in error_msg.lower() for keyword in [
                    "connection", "подключение", "refused", "отверг", "10061",
                    "session", "chrome", "webdriver"
                ])
                
                if is_connection_error and attempt < max_retries:
                    # При ошибке соединения пересоздаем драйвер
                    logger.warning(f"[!] Ошибка соединения с драйвером (попытка {attempt + 1}/{max_retries + 1}): {error_msg[:100]}")
                    if self.driver:
                        try:
                            self.driver.quit()
                        except Exception:
                            pass
                        self.driver = None
                    # Небольшая задержка перед повторной попыткой
                    time.sleep(2)
                    continue
                else:
                    logger.error(f"[!] Не удалось загрузить {url}: {error_msg}")
                    return None
        
        return None


# =============================
# AI analyzer (GigaChat)
# =============================
class LeasingAssetAnalyzer:
    def __init__(self, gigachat_auth_data: str, fetcher: SeleniumFetcher):
        self.gigachat_auth_data = gigachat_auth_data
        self.fetcher = fetcher
        self.access_token = None
        self.token_expires_at = 0

    def _get_gigachat_token(self) -> Optional[str]:
        now = time.time()
        if self.access_token and now < self.token_expires_at:
            return self.access_token

        url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {self.gigachat_auth_data}",
        }
        payload = {"scope": "GIGACHAT_API_PERS"}
        try:
            resp = requests.post(url, headers=headers, data=payload, verify=False, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            self.access_token = data["access_token"]
            self.token_expires_at = data.get("expires_at", 0) / 1000 or now + 1700
            return self.access_token
        except Exception as exc:
            print(f"[!] Ошибка авторизации GigaChat: {exc}")
            return None

    def clean_content(self, html_content: str) -> str:
        if not html_content:
            return ""
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript", "aside", "form", "button"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        # Уменьшаем лимит с 10000 до 5000 символов
        return text[:5000]

    def analyze_with_ai(self, text_content: str) -> Optional[dict]:
        if not text_content:
            return None
        token = self._get_gigachat_token()
        if not token:
            return None

        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }
        system_prompt = """Ты — эксперт-аналитик лизингового оборудования, специализирующийся на анализе объявлений.
Твоя задача — обработать описание автомобиля или техники и выдать структурированное заключение.
Следуй следующим правилам:

Обязательные поля:
1. Укажи категорию (Category).
2. Назови марку и модель автомобиля/техники.
3. Выдели 3–5 ключевых технических характеристик (Specs), которые важны для оценки стоимости и состояния. Примеры: мощность в л.с./кВт; пробег в км/моточасах; объём двигателя в литрах.
4. Перечисли 3–5 преимуществ (Pros) и недостатков/рисков (Cons).
5. Укажи все упомянутые аналоги конкурентов (например, «аналог Volvo...»), если таковые есть.

Формат ответа (валидный JSON):
{
  "category": "string (пример: 'Легковые автомобили', 'Строительная техника')",
  "vendor": "string (производитель, например: 'КамАЗ', 'BMW')",
  "model": "string (точная модель, например: 'КАМАЗ-6522', 'X5 M')",
  "price": int (цена в рублях, null если указано 'не указано'),
  "currency": "string (RUB, USD, EUR)",
  "monthly_payment": int (платёж в рублях, null если не указан),
  "year": int (год выпуска),
  "condition": "string (новый / с пробегом / восстановленный)",
  "location": "string (город/регион)",
  "specs": {
    "техническая_характеристика_1": "значение",
    "техническая_характеристика_2": "значение",
    "техническая_характеристика_3": "значение"
  },
  "pros": ["плюс 1", "плюс 2"],
  "cons": ["минус 1", "минус 2"],
  "analogs_mentioned": ["аналог 1", "аналог 2"]
}
Верни только валидный JSON без markdown-обёртки."""

        payload = {
            "model": "GigaChat-2",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_content},
            ],
            "temperature": 0.1,
            "max_tokens": 1500,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, verify=False, timeout=40)
            resp.raise_for_status()
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
            return safe_json_loads(content)
        except Exception as exc:
            print(f"[!] GigaChat: {exc}")
            return None

    def fetch_page(self, url: str, scroll_times: int = 2, wait: float = 1.5) -> Optional[str]:
        return self.fetcher.fetch_page(url, scroll_times=scroll_times, wait=wait)

    def suggest_analogs(self, item_name: str) -> list[str]:
        """Ask LLM for close analog models when текст объявления недоступен или данных мало."""
        token = self._get_gigachat_token()
        if not token:
            return []
        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }
        system_prompt = (
            "Ты подбираешь аналоги оборудования/техники. "
            "На входе строка с предметом (марка/модель). "
            "Верни JSON {\"analogs\": [\"модель1\", \"модель2\", ...]} максимум 5 элементов."
        )
        payload = {
            "model": "GigaChat-2",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item_name},
            ],
            "temperature": 0.2,
            "max_tokens": 500,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, verify=False, timeout=20)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            data = safe_json_loads(content) or {}
            return ensure_list_str(data.get("analogs"))
        except Exception as exc:
            print(f"[!] Не удалось запросить аналоги: {exc}")
            return []

    def review_analog(self, analog_name: str, listings: list[dict]) -> dict:
        """
        Получить краткий анализ аналога: ключевые плюсы/минусы и примерную цену на основе найденных объявлений.
        """
        token = self._get_gigachat_token()
        if not token:
            return {}
        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }
        listings_text = "\n".join(
            f"- {l.get('title','')} ({l.get('link','')}) {l.get('snippet','')}"
            for l in listings
        )
        system_prompt = (
            "Ты сравниваешь модели техники и формируешь лаконичный обзор.\n"
            "На входе название аналога и несколько заголовков объявлений.\n"
            "Верни JSON с ключами: pros (<=3), cons (<=3), price_hint (int|null), note (str), best_link (str|null)."
        )
        user_content = f"Модель: {analog_name}\nОбъявления:\n{listings_text}"
        payload = {
            "model": "GigaChat-2",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
            "max_tokens": 600,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, verify=False, timeout=20)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return safe_json_loads(content) or {}
        except Exception as exc:
            print(f"[!] Не удалось проанализировать аналог {analog_name}: {exc}")
            return {}

    def validate_report(self, report: dict) -> dict:
        """
        Ask LLM to sanity check the market report.
        """
        token = self._get_gigachat_token()
        if not token:
            return {"is_valid": True, "comment": "AI not available"}

        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }

        # Simplified report for validation
        summary_report = {
            "item": report.get("item"),
            "median_price": report.get("median_price"),
            "mean_price": report.get("mean_price"),
            "market_range": report.get("market_range"),
            "offers_count": len(report.get("offers_used", [])),
        }
        details = json.dumps(summary_report, ensure_ascii=False, default=str)
        system_prompt = (
            "Ты — финансовый аналитик. Проверь рыночную оценку.\n"
            "Если цена кажется адекватной для указанного оборудования, верни 'is_valid': true.\n"
            "Если цена на порядок отличается от реальности (например, буровая за 100 рублей или телефон за 1млрд), 'is_valid': false.\n"
            "Верни JSON: {\"is_valid\": bool, \"comment\": \"reason\"}"
        )

        payload = {
            "model": "GigaChat-2",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Отчет:\n{details}"},
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, verify=False, timeout=20)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return safe_json_loads(content) or {"is_valid": True, "comment": "Parse error"}
        except Exception as e:
            print(f"[!] Validation warning: {e}")
            return {"is_valid": True}

    def rank_offers_by_relevance(self, item_name: str, offers: list[LeasingOffer]) -> list[tuple[LeasingOffer, float]]:
        """
        Оценивает релевантность оферт относительно запроса через GigaChat.
        Возвращает список кортежей (offer, relevance_score), отсортированных по убыванию релевантности.
        """
        if not offers:
            return []
        
        token = self._get_gigachat_token()
        if not token:
            # Если нет токена, возвращаем оферты без сортировки с нейтральным score
            return [(offer, 0.5) for offer in offers]
        
        # Формируем промпт для оценки релевантности
        offers_text = "\n".join([
            f"{i+1}. {offer.title} | {offer.source} | {offer.price_str or 'Цена не указана'} | {offer.url}"
            for i, offer in enumerate(offers)
        ])
        
        system_prompt = """Ты — эксперт по оценке релевантности объявлений о лизинге.
Твоя задача — оценить, насколько каждое объявление соответствует запросу пользователя.
Оцени релевантность каждого объявления по шкале от 0.0 до 1.0, где:
- 1.0 — идеальное соответствие (точная модель, год, характеристики)
- 0.8-0.9 — очень хорошее соответствие (та же модель, близкий год)
- 0.6-0.7 — хорошее соответствие (аналог или похожая модель)
- 0.4-0.5 — среднее соответствие (похожая категория)
- 0.0-0.3 — низкое соответствие (другая категория или нерелевантно)

Верни только JSON массив чисел (scores) в том же порядке, что и объявления.
Пример: [0.95, 0.87, 0.72, 0.65, 0.45]"""
        
        user_prompt = f"""Запрос пользователя: {item_name}

Объявления для оценки:
{offers_text}

Верни массив оценок релевантности (от 0.0 до 1.0) для каждого объявления в том же порядке."""
        
        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }
        
        payload = {
            "model": "GigaChat-2",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, verify=False, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
            
            # Парсим JSON из ответа
            scores = safe_json_loads(content)
            
            # Убеждаемся, что количество scores совпадает с количеством оферт
            if isinstance(scores, list) and len(scores) == len(offers):
                # Нормализуем scores в диапазон [0, 1]
                scores = [max(0.0, min(1.0, float(s))) for s in scores]
                return list(zip(offers, scores))
            else:
                logger.warning(f"[!] GigaChat вернул неверное количество оценок: {len(scores) if isinstance(scores, list) else 0} вместо {len(offers)}")
                return [(offer, 0.5) for offer in offers]
        except Exception as e:
            logger.error(f"[!] Ошибка при оценке релевантности через GigaChat: {e}")
            return [(offer, 0.5) for offer in offers]


# =============================
# Basic regex parsing for non-Avito pages
# =============================
def parse_page_basic(html_content: str, model_name: str) -> dict:
    result = {}
    if not html_content:
        return result
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)

    if re.search(r"цена\s+по\s+запросу|по\s+договоренности", text, flags=re.IGNORECASE):
        result["price_on_request"] = True

    display_price = extract_price_candidate(text)
    if display_price:
        result["price"] = display_price

    year = _extract_year_from_text(text)
    if year:
        result["year"] = year

    power = _extract_power(text)
    if power:
        result["power"] = power

    mileage = _extract_mileage(text)
    if mileage:
        result["mileage"] = mileage

    # vendor heuristic from model
    if model_name:
        result.setdefault("vendor", model_name.split()[0])

    return result


# =============================
# Search and URL handling
# =============================
def search_google(query: str, num_results: int = 10, specialized: bool = False) -> list[dict]:
    """
    Поиск через Google с поддержкой специализированных запросов.
    
    Args:
        query: Поисковый запрос
        num_results: Количество результатов
        specialized: Если True, использует более специфичные запросы для промышленного оборудования
    """
    if not SERPER_API_KEY:
        print("[!] SERPER_API_KEY не задан.")
        return []
    
    # Для специализированного оборудования добавляем специфичные термины
    if specialized:
        if is_specialized_equipment(query):
            # Добавляем специфичные термины для лучшего поиска
            query = f"{query} лизинг промышленного оборудования OR аренда OR продажа"
    
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={
                "q": query, 
                "gl": "ru", 
                "hl": "ru", 
                "num": num_results,
                # Исключаем популярные сайты, если ищем специализированное оборудование
                "excludeDomains": "avito.ru,drom.ru,auto.ru" if specialized else None,
            },
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("organic", [])
        
        # Приоритизируем специализированные домены
        if specialized:
            specialized_domains = [
                "promleasing", "techleasing", "equipment", "leasing",
                "пром", "тех", "оборудование", "лизинг"
            ]
            # Сортируем: сначала специализированные
            results.sort(key=lambda x: any(
                domain in urlparse(x.get("link", "")).netloc.lower() 
                for domain in specialized_domains
            ), reverse=True)
        
        return results
    except Exception as exc:
        print(f"[!] Ошибка поиска: {exc}")
        return []


# Расширенный список источников для специализированного оборудования
MANDATORY_SOURCES = [
    {
        "name": "alfaleasing.ru",
        "search_url": "https://alfaleasing.ru/search/?q={query}",
    },
    {
        "name": "sberleasing.ru",
        "search_url": "https://www.sberleasing.ru/search/?q={query}",
    },
    {
        "name": "avito.ru",
        "search_url": "https://www.avito.ru/rossiya?q={query}+лизинг",
    },
]

# Специализированные источники для промышленного оборудования
SPECIALIZED_SOURCES = [
    {
        "name": "promleasing.ru",
        "search_url": "https://promleasing.ru/search?q={query}",
        "categories": ["промышленное", "буровое", "строительное"]
    },
    {
        "name": "techleasing.ru",
        "search_url": "https://techleasing.ru/catalog?search={query}",
        "categories": ["техника", "оборудование"]
    },
    {
        "name": "equipment-leasing.ru",
        "search_url": "https://equipment-leasing.ru/search/{query}",
        "categories": ["оборудование", "техника"]
    },
    # Добавьте другие специализированные сайты
]

def generate_mandatory_urls(model_name: str, include_specialized: bool = True) -> list[dict]:
    query_encoded = model_name.replace(" ", "+").lower()
    mandatory = []
    
    # Обычные источники
    for source in MANDATORY_SOURCES:
        url = source["search_url"].format(query=query_encoded)
        mandatory.append({
            "link": url,
            "title": f"{model_name} - {source['name']}",
            "is_mandatory": True,
            "source_name": source["name"],
        })
    
    # Специализированные источники (если включены)
    if include_specialized:
        for source in SPECIALIZED_SOURCES:
            url = source["search_url"].format(query=query_encoded)
            mandatory.append({
                "link": url,
                "title": f"{model_name} - {source['name']}",
                "is_mandatory": True,
                "source_name": source["name"],
                "is_specialized": True,
            })
    
    return mandatory


def filter_search_results(results: list[dict], max_results: int = 10, specialized: bool = False) -> list[dict]:
    """
    Фильтрует результаты поиска с приоритетом специализированных сайтов.
    """
    filtered = []
    blocked_domains = {"chelindleasing"}
    
    # Домены специализированных сайтов (высокий приоритет)
    specialized_domains = {
        "promleasing", "techleasing", "equipment-leasing", 
        "промлизинг", "техлизинг", "оборудование",
        "alfaleasing", "sberleasing"
    }
    
    # Популярные сайты (низкий приоритет для специализированного оборудования)
    popular_domains = {"avito", "drom", "auto.ru", "youla"}
    
    # Разделяем на категории
    specialized_results = []
    regular_results = []
    popular_results = []
    
    for result in results:
        if len(filtered) >= max_results * 2:  # Собираем больше для сортировки
            break
        url = result.get("link", "")
        domain = urlparse(url).netloc.replace("www.", "").lower()
        
        if any(blocked in domain for blocked in blocked_domains):
            continue
        
        if specialized and any(spec_domain in domain for spec_domain in specialized_domains):
            specialized_results.append(result)
        elif specialized and any(pop_domain in domain for pop_domain in popular_domains):
            popular_results.append(result)  # Низкий приоритет
        else:
            regular_results.append(result)
    
    # Объединяем: сначала специализированные, потом обычные, потом популярные
    if specialized:
        filtered = specialized_results + regular_results + popular_results
    else:
        filtered = regular_results + specialized_results + popular_results
    
    return filtered[:max_results]


def merge_with_mandatory(search_results: list[dict], mandatory: list[dict]) -> list[dict]:
    existing_domains = {urlparse(r.get("link", "")).netloc.replace("www.", "") for r in search_results}
    merged = []
    for m in mandatory:
        domain = m.get("source_name", "")
        if domain not in existing_domains:
            merged.append(m)
            existing_domains.add(domain)
    merged.extend(search_results)
    return merged


# =============================
# Market analysis
# =============================
def percentile(sorted_values: list[int], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return float(sorted_values[int(k)])
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return float(d0 + d1)


def filter_price_outliers(offers: list[LeasingOffer]) -> list[LeasingOffer]:
    prices = [o.price for o in offers if o.price is not None]
    if len(prices) < 5:
        return offers
    prices_sorted = sorted(prices)
    q1 = percentile(prices_sorted, 0.25)
    q3 = percentile(prices_sorted, 0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filtered = [o for o in offers if o.price is None or (lower <= o.price <= upper)]
    removed = len(offers) - len(filtered)
    if removed:
        print(f"[*] Удалено выбросов: {removed}")
    return filtered


def collect_analogs(item_name: str, offers: list[LeasingOffer], use_ai: bool, analyzer: Optional[LeasingAssetAnalyzer]) -> list[str]:
    analogs_set = set()
    for o in offers:
        for a in o.analogs:
            analogs_set.add(a.strip())
    # If not enough analogs, try AI suggestion
    if len(analogs_set) < 3 and use_ai and analyzer:
        ai_analogs = analyzer.suggest_analogs(item_name)
        for a in ai_analogs:
            analogs_set.add(a)
    # Fallback: simple search titles for "аналог" results
    if len(analogs_set) < 3:
        fallback_results = search_google(f"{item_name} аналог", 5)
        for r in fallback_results:
            title = r.get("title") or ""
            parts = re.split(r"[–—|-]", title)
            if parts:
                candidate = parts[0].strip()
                if candidate and len(candidate.split()) <= 6:
                    analogs_set.add(candidate)
    return [a for a in analogs_set if a]


def fetch_listing_summaries(query: str, top_n: int = 3) -> list[dict]:
    results = search_google(query, num_results=top_n)
    summaries = []
    for r in results[:top_n]:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        # Use smarter extraction
        price_guess = extract_price_candidate(title) or extract_price_candidate(snippet)
        summaries.append(
            {
                "title": title,
                "link": r.get("link", ""),
                "snippet": snippet,
                "price_guess": price_guess,
            }
        )
    return summaries


def analyze_market(item_name: str, offers: list[LeasingOffer], client_price: Optional[int]) -> dict:
    prices = [o.price for o in offers if o.price is not None]
    # Подсчитываем оферты с ценой по запросу
    offers_with_price_on_request = [o for o in offers if o.price_on_request and o.price is None]
    offers_with_price = [o for o in offers if o.price is not None]
    
    result = {
        "item": item_name,
        "offers_used": [asdict(o) for o in offers],
        "analogs_suggested": [],
        "market_range": None,
        "median_price": None,
        "mean_price": None,
        "client_price": client_price,
        "client_price_ok": None,
        "explanation": "",
        "offers_with_price_count": len(offers_with_price),
        "offers_price_on_request_count": len(offers_with_price_on_request),
    }
    
    # Если есть оферты с ценой по запросу, но нет цен - сообщаем об этом
    if not prices and offers_with_price_on_request:
        result["explanation"] = (
            f"Найдено {len(offers)} объявлений, из них {len(offers_with_price_on_request)} с ценой по запросу. "
            f"Для расчета рыночной стоимости недостаточно данных о ценах."
        )
        return result
    
    if not prices:
        result["explanation"] = "Нет собранных цен."
        return result

    prices_sorted = sorted(prices)
    min_p, max_p = prices_sorted[0], prices_sorted[-1]
    
    # Преобразуем цены в float перед вычислением статистики, чтобы избежать переполнения
    try:
        prices_float = [float(p) for p in prices_sorted]
        median_p = statistics.median(prices_float)
        mean_p = statistics.mean(prices_float)
        # Округляем mean_p до int для консистентности
        mean_p = round(mean_p)
    except (OverflowError, ValueError) as e:
        # Если произошло переполнение, используем альтернативный метод
        logger.warning(f"[!] Ошибка при вычислении статистики: {e}, используем упрощенный расчет")
        # Для медианы берем средний элемент
        n = len(prices_sorted)
        if n % 2 == 0:
            median_p = (prices_sorted[n//2 - 1] + prices_sorted[n//2]) / 2.0
        else:
            median_p = float(prices_sorted[n//2])
        # Для среднего используем деление с float
        mean_p = round(sum(prices_sorted) / float(len(prices_sorted)))

    result["market_range"] = [min_p, max_p]
    result["median_price"] = float(median_p) if median_p is not None else None
    result["mean_price"] = float(mean_p) if mean_p is not None else None

    # Формируем explanation с учетом оферт с ценой по запросу
    price_info = f"Рыночный диапазон {format_price(min_p)} – {format_price(max_p)}, медиана {format_price(int(median_p))}."
    if offers_with_price_on_request:
        price_info += f" Также найдено {len(offers_with_price_on_request)} объявлений с ценой по запросу."

    if client_price is not None:
        try:
            deviation = (client_price - median_p) / median_p * 100
            ok = abs(deviation) <= 20  # 20% tolerance
            result["client_price_ok"] = ok
            verdict = "подтверждена" if ok else "не подтверждена"
            result["explanation"] = (
                f"{price_info} "
                f"Цена клиента {format_price(client_price)} ({deviation:+.1f}%), {verdict}."
            )
        except (OverflowError, ZeroDivisionError) as e:
            logger.warning(f"[!] Ошибка при вычислении отклонения: {e}")
            result["client_price_ok"] = None
            result["explanation"] = (
                f"{price_info} "
                f"Цена клиента {format_price(client_price)}."
            )
    else:
        result["explanation"] = price_info

    return result


# =============================
# Pipeline
# =============================
def extract_model_from_query(query: str) -> str:
    parts = query.split()
    index = parts.index('купить')
    return " ".join(parts[:index]) if parts else ""


def search_and_analyze(
    query: str,
    fetcher: SeleniumFetcher,
    analyzer: Optional[LeasingAssetAnalyzer],
    num_results: int = 5,
    use_ai: bool = True,
    specialized: bool = False,  # Новый параметр
) -> list[LeasingOffer]:
    print("\n" + "=" * 70)
    print(f"Поисковый запрос: {query}")
    if specialized:
        print("[*] Режим поиска специализированного оборудования")
    print("=" * 70)

    model_name = extract_model_from_query(query)
    mandatory_urls = generate_mandatory_urls(model_name, include_specialized=specialized)
    print(f"[*] Обязательные источники: {len(mandatory_urls)}")

    # Используем улучшенный поиск
    search_results = search_google(query, num_results * 2, specialized=specialized)  # Больше результатов для фильтрации
    filtered_google = filter_search_results(search_results, num_results * 2, specialized=specialized) if search_results else []

    all_results = merge_with_mandatory(filtered_google, mandatory_urls)
    print(f"[*] Всего URL: {len(all_results)}")

    offers: list[LeasingOffer] = []

    for idx, result in enumerate(all_results, 1):
        url = result.get("link", "")
        title = result.get("title", "")
        domain = urlparse(url).netloc.replace("www.", "")
        is_avito = "avito.ru" in domain

        print(f"\n[{idx}/{len(all_results)}] {domain} | {url}")

        # Fetch page with tuned scroll depth
        scroll_times = 2 if is_avito else 3
        html = fetcher.fetch_page(url, scroll_times=scroll_times, wait=1.5)
        if not html:
            print("    [!] Не удалось загрузить страницу")
            continue

        if is_avito:
            avito_offers = parse_avito_list_page(html, model_name)
            print(f"    Объявлений Avito: {len(avito_offers)}")
            offers.extend(avito_offers)
            continue  # no AI for Avito list pages

        # Basic parsing
        basic = parse_page_basic(html, model_name)

        ai_result = None
        if use_ai and analyzer:
            text = analyzer.clean_content(html)
            ai_result = analyzer.analyze_with_ai(text)

        merged = basic.copy()
        if ai_result:
            for k, v in ai_result.items():
                if v is not None:
                    merged[k] = v

        if not merged:
            print("    [!] Нет структурированных данных")
            continue

        offer = LeasingOffer(
            title=title or result.get("title", "Объявление"),
            url=url,
            source=domain,
            model=model_name,
            price=merged.get("price"),
            price_str=format_price(merged.get("price")),
            monthly_payment=merged.get("monthly_payment"),
            monthly_payment_str=format_price(merged.get("monthly_payment")),
            price_on_request=merged.get("price_on_request", False),
            year=merged.get("year"),
            power=merged.get("power"),
            mileage=merged.get("mileage"),
            vendor=merged.get("vendor"),
            condition=merged.get("condition"),
            location=merged.get("location"),
            specs=merged.get("specs", {}),
            category=merged.get("category"),
            currency=merged.get("currency"),
            pros=ensure_list_str(merged.get("pros")),
            cons=ensure_list_str(merged.get("cons")),
            analogs=ensure_list_str(merged.get("analogs_mentioned")),
        )
        if offer.has_data():
            offers.append(offer)

    offers = filter_price_outliers(offers)
    return offers


def print_results(offers: list[LeasingOffer]):
    print("\n" + "=" * 70)
    print(f"Найдено предложений: {len(offers)}")
    print("=" * 70)
    for i, o in enumerate(offers, 1):
        print(f"\n[{i}] {o.title}")
        print(f"    Источник: {o.source}")
        print(f"    Модель: {o.model}")
        if o.category:
            print(f"    Категория: {o.category}")
        print(f"    URL: {o.url}")
        if o.price_str or o.monthly_payment_str or o.price_on_request:
            print("    --- Ценообразование ---")
            if o.price_on_request and not o.price:
                print("    Цена по запросу")
            if o.price_str:
                if o.currency and o.currency.upper() != "RUB":
                    print(f"    Цена: {o.price_str} ({o.currency})")
                else:
                    print(f"    Цена: {o.price_str}")
            if o.monthly_payment_str:
                print(f"    Месячный платеж: {o.monthly_payment_str}")
        if any([o.year, o.power, o.mileage, o.vendor, o.condition, o.location]):
            print("    --- Характеристики ---")
            if o.vendor:
                print(f"    Производитель: {o.vendor}")
            if o.year:
                print(f"    Год: {o.year}")
            if o.condition:
                print(f"    Состояние: {o.condition}")
            if o.power:
                print(f"    Мощность: {o.power}")
            if o.mileage:
                print(f"    Пробег: {o.mileage}")
            if o.location:
                print(f"    Локация: {o.location}")
        if o.specs:
            print("    --- Доп. параметры ---")
            for k, v in o.specs.items():
                print(f"    {k}: {v}")
        if o.pros:
            print("    --- Плюсы ---")
            for p in o.pros:
                print(f"    + {p}")
        if o.cons:
            print("    --- Минусы ---")
            for c in o.cons:
                print(f"    - {c}")
        if o.analogs:
            print("    --- Упомянутые аналоги ---")
            for a in o.analogs:
                print(f"    • {a}")


def save_results_json(offers: list[LeasingOffer], item_name: str = "results", market_report: Optional[dict] = None):
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in item_name)
    filename = f"{safe_name}.json"
    data = {"offers": [asdict(o) for o in offers]}
    if market_report:
        data["market_report"] = market_report
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n[*] JSON сохранен в {filename}")


# =============================
# CLI
# =============================
def main():
    print("=" * 70)
    print("Анализатор рыночной стоимости предмета лизинга (Avito + AI)")
    print("=" * 70)

    item = input("\nВведите предмет лизинга (например, BMW M5 2024): ").strip()
    if not item:
        print("[!] Пустой запрос, выходим.")
        return

    client_price_input = input("Цена клиента (только цифры, опционально): ").strip()
    client_price = digits_to_int(client_price_input) if client_price_input else None

    # Ask for AI usage. Default: Y
    use_ai_str = input("Использовать AI для анализа (y/n, default y): ").strip().lower()
    use_ai = use_ai_str != "n"

    num_input = input("Сколько результатов искать (default 5): ").strip()
    num_results = int(num_input) if num_input.isdigit() else 5

    # Lifecycle management
    fetcher = SeleniumFetcher()
    analyzer = LeasingAssetAnalyzer(GIGACHAT_AUTH_DATA, fetcher) if use_ai else None

    try:
        query = f"{item} лизинг"
        offers = search_and_analyze(query, fetcher, analyzer, num_results=num_results, use_ai=use_ai)
        
        # Retry logic if empty
        if not offers:
            print("\n[!] Прямой поиск не дал результатов, пробуем упростить запрос...")
            # Try searching just the item name
            query_simple = f"{item} купить"
            offers = search_and_analyze(query_simple, fetcher, analyzer, num_results=num_results, use_ai=use_ai)
        
        if not offers:
            print("\n[!] Не удалось извлечь предложения даже после повторной попытки.")
            return

        print_results(offers)
        
        # Analogs
        analogs = collect_analogs(item, offers, use_ai=use_ai, analyzer=analyzer)
        
        # Generate initial report
        report = analyze_market(item, offers, client_price)
        report["analogs_suggested"] = analogs

        # Validate with AI
        if use_ai and analyzer and report["median_price"]:
            print("\n[*] Проверка отчета через AI...")
            validation = analyzer.validate_report(report)
            if not validation.get("is_valid"):
                print(f"[WARN] AI считает отчет подозрительным: {validation.get('comment')}")
                report["ai_flag"] = "SUSPICIOUS"
                report["ai_comment"] = validation.get("comment")
            else:
                print("[*] AI подтверждает корректность оценки.")

        # Analogs deep dive
        analog_details = []
        if analogs:
            print("\n[*] Сбор данных по аналогам...")
            for analog in analogs:
                listings = fetch_listing_summaries(f"{analog} купить", top_n=3)
                price_list = [l["price_guess"] for l in listings if l.get("price_guess")]
                avg_price_math = int(sum(price_list) / len(price_list)) if price_list else None
                
                pros, cons, note = [], [], ""
                price_hint = None
                best_link = None

                if use_ai and analyzer:
                    ai_review = analyzer.review_analog(analog, listings)
                    pros = ensure_list_str(ai_review.get("pros"))
                    cons = ensure_list_str(ai_review.get("cons"))
                    price_hint = ai_review.get("price_hint")
                    note = ai_review.get("note", "")
                    best_link = ai_review.get("best_link")
                
                # Prioritize AI price hint if available and reasonable, else math
                final_price = price_hint if price_hint else avg_price_math

                analog_details.append(
                    {
                        "name": analog,
                        "listings": listings,
                        "avg_price_guess": final_price,
                        "ai_price_hint": price_hint,
                        "pros": pros,
                        "cons": cons,
                        "note": note,
                        "best_link": best_link
                    }
                )

        report["analogs_details"] = analog_details

        # Final print
        print("\n" + "=" * 70)
        print("ИТОГОВЫЙ ОТЧЕТ")
        print("=" * 70)
        if report["market_range"]:
             min_p, max_p = report["market_range"]
             print(f"Диапазон рынка: {format_price(min_p)} – {format_price(max_p)}")
             print(f"Медиана: {format_price(report['median_price'])}")
        
        if client_price:
            status = "OK" if report.get("client_price_ok") else "Deviation > 20%"
            print(f"Цена клиента: {format_price(client_price)} -> {status}")
        
        print(f"Комментарий (стат): {report['explanation']}")
        
        if report.get("ai_flag"):
            print(f"WARNING: {report.get('ai_comment')}")

        if analog_details:
            print("\nСравнение с аналогами:")
            for a in analog_details:
                p_est = a.get("avg_price_guess")
                print(f"--- {a['name']} ---")
                print(f"  Цена ~ {format_price(p_est) if p_est else 'Нет данных'}")
                if a.get('note'): print(f"  Заметка: {a['note']}")
                if a['pros']: print(f"  [+] {', '.join(a['pros'])}")
                if a['cons']: print(f"  [-] {', '.join(a['cons'])}")
                
                # Print sources clearly
                print("  Источники:")
                printed_links = set()
                # If AI highlighted a best link, show it first
                if a.get("best_link"):
                    print(f"    [Реком.] {a['best_link']}")
                    printed_links.add(a['best_link'])
                
                # Show other listings
                if a.get("listings"):
                    for l in a["listings"]:
                        lnk = l.get('link','')
                        if lnk and lnk not in printed_links:
                            print(f"    {l.get('title','Link')}: {lnk}")

        save_input = input("\nСохранить результаты в JSON? (y/n): ").strip().lower()
        if save_input == "y":
            save_results_json(offers, item, market_report=report)

    finally:
        if fetcher:
            fetcher.close()

# parser.py

def run_analysis(
    item: str,
    client_price: int | None = None,
    use_ai: bool = True,
    num_results: int = 5,
) -> dict:
    """
    Высокоуровневая точка входа для API.
    """
    if not item:
        return {"error": "empty_item"}

    fetcher = SeleniumFetcher()
    analyzer = LeasingAssetAnalyzer(GIGACHAT_AUTH_DATA, fetcher) if use_ai else None

    try:
        query = f"{item} лизинг"
        offers = search_and_analyze(query, fetcher, analyzer, num_results=num_results, use_ai=use_ai)

        if not offers:
            # fallback как в CLI
            query_simple = f"{item} купить"
            offers = search_and_analyze(query_simple, fetcher, analyzer, num_results=num_results, use_ai=use_ai)

        if not offers:
            return {
                "item": item,
                "offers_used": [],
                "analogs_suggested": [],
                "analogs_details": [],
                "market_report": {
                    "item": item,
                    "market_range": None,
                    "median_price": None,
                    "mean_price": None,
                    "client_price": client_price,
                    "client_price_ok": None,
                    "explanation": "Не удалось собрать предложения.",
                },
            }

        # базовый отчёт
        report = analyze_market(item, offers, client_price)
        analogs = collect_analogs(item, offers, use_ai=use_ai, analyzer=analyzer)
        report["analogs_suggested"] = analogs

        # валидация AI
        if use_ai and analyzer and report.get("median_price"):
            validation = analyzer.validate_report(report)
            if not validation.get("is_valid"):
                report["ai_flag"] = "SUSPICIOUS"
                report["ai_comment"] = validation.get("comment")
            else:
                report["ai_flag"] = "OK"

        # подробности по аналогам (упрощённо)
        analog_details = []
        for analog in analogs:
            listings = fetch_listing_summaries(f"{analog} купить", top_n=3)
            price_list = [l["price_guess"] for l in listings if l.get("price_guess")]
            avg_price_math = int(sum(price_list) / len(price_list)) if price_list else None

            pros, cons, note = [], [], ""
            price_hint = None
            best_link = None

            if use_ai and analyzer:
                ai_review = analyzer.review_analog(analog, listings) or {}
                pros = ensure_list_str(ai_review.get("pros"))
                cons = ensure_list_str(ai_review.get("cons"))
                price_hint = ai_review.get("price_hint")
                note = ai_review.get("note", "")
                best_link = ai_review.get("best_link")

            final_price = price_hint if price_hint else avg_price_math

            analog_details.append(
                {
                    "name": analog,
                    "listings": listings,
                    "avg_price_guess": final_price,
                    "ai_price_hint": price_hint,
                    "pros": pros,
                    "cons": cons,
                    "note": note,
                    "best_link": best_link,
                }
            )

        report["analogs_details"] = analog_details

        return {
            "item": item,
            "offers_used": [asdict(o) for o in offers],
            "analogs_suggested": analogs,
            "analogs_details": analog_details,
            "market_report": report,
        }
    finally:
        fetcher.close()


if __name__ == "__main__":
    main()
