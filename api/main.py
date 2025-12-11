from typing import List, Optional

import os
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
)  # [file:42]

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
    item: str = Field(..., example="MAN TGS 26.440 2018", description="Что анализируем")
    client_price: Optional[float] = Field(
        None,
        example=10500000,
        description="Цена клиента в рублях (опционально)",
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
        # Первый поисковый запрос: "<item> лизинг"
        query = f"{req.item} лизинг"

        # search_and_analyze — функция из parser_b.py:
        # - делает поиск (Serper + обязательные источники),
        # - парсит страницы Selenium'ом,
        # - запускает AI, если use_ai = True,
        # - фильтрует выбросы по ценам,
        # - возвращает список LeasingOffer.
        offers = search_and_analyze(
            query,
            fetcher=fetcher,
            analyzer=analyzer,
            num_results=req.num_results,
            use_ai=use_ai,
        )

        # Если по запросу "лизинг" ничего не нашли — пробуем «купить».
        if not offers:
            query_simple = f"{req.item} купить"
            offers = search_and_analyze(
                query_simple,
                fetcher=fetcher,
                analyzer=analyzer,
                num_results=req.num_results,
                use_ai=use_ai,
            )

        # Совсем пусто — отдаём 404, фронт покажет ошибку.
        if not offers:
            raise HTTPException(status_code=404, detail="Не удалось найти предложения")

        # Статистический анализ рынка:
        # analyze_market уже реализован в parser_b.py и возвращает dict
        # с полями: market_range, median_price, client_price_ok, explanation и т.д.
        report = analyze_market(req.item, offers, client_price)

        # Преобразуем внутренние объекты LeasingOffer
        # в Pydantic-модели OfferDTO, чтобы FastAPI вернул корректный JSON.
        offers_dto: List[OfferDTO] = []
        for o in report["offers_used"]:
            if isinstance(o, dict):
                offers_dto.append(OfferDTO(**o))
            else:
                # dataclass LeasingOffer -> используем __dict__
                offers_dto.append(OfferDTO(**o.__dict__))

        # Собираем объект отчёта для ответа
        report_dto = MarketReportDTO(
            item=report["item"],
            offers_used=offers_dto,
            analogs_suggested=report.get("analogs_suggested", []),
            market_range=report.get("market_range"),
            median_price=report.get("median_price"),
            mean_price=report.get("mean_price"),
            client_price=report.get("client_price"),
            client_price_ok=report.get("client_price_ok"),
            explanation=report.get("explanation", ""),
            ai_flag=report.get("ai_flag"),
            ai_comment=report.get("ai_comment"),
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
