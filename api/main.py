from typing import List, Optional

import os
import sys
import logging

# Добавляем родительскую директорию в sys.path для импорта parser_b
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Импортируем ваш парсер и аналитику из parser_b.py
from parser_b import (
    SeleniumFetcher,
    LeasingAssetAnalyzer,
    GIGACHAT_AUTH_DATA,
    search_and_analyze,
    analyze_market,
)

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Создаём FastAPI-приложение
app = FastAPI(title="Leasing Asset Market Analyzer API")

BASE_DIR = os.path.dirname(__file__)
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=FileResponse)
def root():
    return FileResponse(os.path.join(templates_dir, "index.html"))

# Включаем CORS, чтобы фронтенд (другой домен/порт) мог вызывать API из браузера
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # на проде лучше указать конкретный домен фронта
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Pydantic-модели запроса/ответа
# =========================

class AnalyzeRequest(BaseModel):
    item: str = Field(
        ..., 
        description="Что анализируем",
        json_schema_extra={"example": "MAN TGS 26.440 2018"}
    )
    client_price: Optional[float] = Field(
        None,
        description="Цена клиента в рублях (опционально)",
        json_schema_extra={"example": 10500000}
    )
    num_results: int = Field(
        5,
        ge=1,
        le=20,
        description="Сколько источников/страниц собирать для анализа",
    )


class OfferDTO(BaseModel):
    """
    Одна найденная рыночная оферта (объявление или предложение лизинга).
    Это «чистый» формат для фронта.
    """
    title: str
    url: str
    source: str
    model: Optional[str] = None
    price: Optional[int] = None
    price_str: Optional[str] = None
    monthly_payment: Optional[int] = None
    monthly_payment_str: Optional[str] = None
    price_on_request: Optional[bool] = None
    year: Optional[int] = None
    power: Optional[str] = None
    mileage: Optional[str] = None
    vendor: Optional[str] = None
    condition: Optional[str] = None
    location: Optional[str] = None
    specs: dict = {}
    category: Optional[str] = None
    currency: Optional[str] = None
    pros: List[str] = []
    cons: List[str] = []
    analogs: List[str] = []
    analogs_suggested: List[str] = []
    relevance_score: Optional[float] = None  # Добавлено: оценка релевантности от 0.0 до 1.0


class MarketReportDTO(BaseModel):
    """
    Итоговый отчёт по рынку, который строится на основе функций из parser_b.py.
    """
    item: str
    offers_used: List[OfferDTO]
    analogs_suggested: List[str]
    market_range: Optional[List[int]] = None    # [min_price, max_price]
    median_price: Optional[float] = None
    mean_price: Optional[float] = None
    client_price: Optional[int] = None
    client_price_ok: Optional[bool] = None
    explanation: str = ""                       # человекочитаемый комментарий
    ai_flag: Optional[str] = None               # например, "SUSPICIOUS"
    ai_comment: Optional[str] = None            # комментарий от AI
    offers_with_price_count: Optional[int] = None  # количество оферт с указанной ценой
    offers_price_on_request_count: Optional[int] = None  # количество оферт с ценой по запросу


class AnalyzeResponse(BaseModel):
    """
    Обёртка для ответа API.
    Можно в будущем добавить сюда поля 'request_id', 'timestamp' и т.д.
    """
    status: str = "ok"
    report: MarketReportDTO


# =========================
# Эндпоинты API
# =========================

@app.get("/health")
def health():
    """
    Простой health-check.
    Можно использовать в мониторинге и для проверки, что сервис жив.
    """
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Основной endpoint:

    1. Принимает описание предмета лизинга и цену клиента.
    2. Запускает пайплайн из parser_b.py:
       - поиск страниц (Google/обязательные источники),
       - парсинг (Avito + другие сайты),
       - AI-анализ (GigaChat) поверх HTML, где нужно.
    3. Считает медиану и диапазон цен.
    4. Возвращает структурированный отчёт для фронта.
    """

    # Приводим цену клиента к int (если она есть)
    client_price = int(req.client_price) if req.client_price is not None else None

    # Создаём Selenium-фетчер (один на запрос)
    fetcher = SeleniumFetcher()

    # На проде AI всегда включён
    use_ai = True

    # Создаём GigaChat-анализатор, который:
    # - получает токен,
    # - чистит HTML,
    # - вытаскивает структуру через LLM.
    analyzer = LeasingAssetAnalyzer(GIGACHAT_AUTH_DATA, fetcher)

    try:
        # 1. Спрашиваем у GigaChat аналоги (до 5 штук)
        analogs = []
        if use_ai:
            try:
                analogs = analyzer.suggest_analogs(req.item)
                # На всякий случай ограничиваем список, если модель вернет больше
                analogs = analogs[:5]
                logger.info(f"[*] GigaChat предложил аналоги: {analogs}")
            except Exception as e:
                logger.error(f"[!] Ошибка при поиске аналогов: {e}")

        # 2. Формируем список запросов: сам предмет + аналоги
        # Используем set для дедупликации названий (на всякий случай), но сохраняем порядок через list
        search_targets = [req.item]
        for a in analogs:
            if a and a.lower() != req.item.lower() and a not in search_targets:
                search_targets.append(a)

        all_offers = []
        seen_urls = set()

        # 3. Проходим поиском по всем целям
        for i, target in enumerate(search_targets):
            logger.info(f"\n--- Обработка цели [{i+1}/{len(search_targets)}]: {target} ---")
            
            # Определяем, является ли это специализированным оборудованием
            from parser_b import is_specialized_equipment
            specialized = is_specialized_equipment(target)
            
            if specialized:
                logger.info(f"[*] Определено специализированное оборудование: {target}")
            
            # Базовый запрос
            query = f"{target} купить в лизинг"
            
            # search_and_analyze с флагом specialized
            offers = search_and_analyze(
                query,
                fetcher=fetcher,
                analyzer=analyzer,
                num_results=req.num_results,
                use_ai=use_ai,
                specialized=specialized,  # Передаем флаг
            )

            # Если это ОСНОВНОЙ предмет и ничего не нашли — пробуем fallback "купить"
            # Для аналогов fallback не делаем, чтобы не раздувать время выполнения
            if not offers and target == req.item:
                logger.info(f"[*] По запросу '{query}' пусто, пробуем 'купить'...")
                query_simple = f"{target} купить"
                offers = search_and_analyze(
                    query_simple,
                    fetcher=fetcher,
                    analyzer=analyzer,
                    num_results=req.num_results,
                    use_ai=use_ai,
                )

            # Агрегируем результаты
            for o in offers:
                if o.url not in seen_urls:
                    seen_urls.add(o.url)
                    all_offers.append(o)

        # Если после всех поисков список пуст — 404
        if not all_offers:
            raise HTTPException(status_code=404, detail="Не удалось найти предложения (ни по предмету, ни по аналогам)")
        
        # Передаем общий список найденного
        offers = all_offers

        # Статистический анализ рынка:
        # analyze_market уже реализован в parser_b.py и возвращает dict
        # с полями: market_range, median_price, client_price_ok, explanation и т.д.
        report = analyze_market(req.item, offers, client_price)
        
        # Добавляем найденные аналоги в отчет, чтобы они вернулись на фронт
        report["analogs_suggested"] = analogs

        # Преобразуем внутренние объекты LeasingOffer
        # в Pydantic-модели OfferDTO, чтобы FastAPI вернул корректный JSON.
        offers_dto: List[OfferDTO] = []
        
        # Сортируем оферты по релевантности через GigaChat
        if use_ai and analyzer and all_offers:
            try:
                logger.info(f"[*] Оценка релевантности {len(all_offers)} оферт через GigaChat...")
                ranked_offers = analyzer.rank_offers_by_relevance(req.item, all_offers)
                # Сортируем по убыванию релевантности
                ranked_offers.sort(key=lambda x: x[1], reverse=True)
                all_offers = [offer for offer, _ in ranked_offers]
                relevance_scores = {offer.url: score for offer, score in ranked_offers}
            except Exception as e:
                logger.error(f"[!] Ошибка при сортировке по релевантности: {e}")
                relevance_scores = {}
        else:
            relevance_scores = {}
        
        for o in report["offers_used"]:
            if isinstance(o, dict):
                offer_dict = dict(o)
                offer_dict["relevance_score"] = relevance_scores.get(o.get("url"), None)
                offers_dto.append(OfferDTO(**offer_dict))
            else:
                # dataclass LeasingOffer -> используем __dict__
                offer_dict = o.__dict__.copy()
                offer_dict["relevance_score"] = relevance_scores.get(o.url, None)
                offers_dto.append(OfferDTO(**offer_dict))

        # Собираем объект отчёта для ответа
        report_dto = MarketReportDTO(
            item=report["item"],
            offers_used=offers_dto,
            analogs_suggested=report.get("analogs_suggested", []),
            market_range=report.get("market_range"),
            median_price=float(report.get("median_price")) if report.get("median_price") is not None else None,
            mean_price=float(report.get("mean_price")) if report.get("mean_price") is not None else None,
            client_price=report.get("client_price"),
            client_price_ok=report.get("client_price_ok"),
            explanation=report.get("explanation", ""),
            ai_flag=report.get("ai_flag"),
            ai_comment=report.get("ai_comment"),
            offers_with_price_count=report.get("offers_with_price_count"),
            offers_price_on_request_count=report.get("offers_price_on_request_count"),
        )

        # Финальный JSON-ответ API
        return AnalyzeResponse(status="ok", report=report_dto)

    except HTTPException:
        # Уже готовая HTTP-ошибка (404/400 и т.п.) — пробрасываем как есть.
        raise
    except Exception as e:
        # Любая неожиданная ошибка — 500.
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # В любом случае корректно закрываем Selenium-драйвер,
        # чтобы не копились процессы Chrome.
        fetcher.close()
