import os
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for parser_b import
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from parser_b import run_analysis

app = FastAPI(
    title="Leasing descriptor API",
    description="Рыночный анализ предмета лизинга + аналоги",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
templates_dir = BASE_DIR / "templates"
static_dir = BASE_DIR / "static"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DescribeRequest(BaseModel):
    text: str
    clientPrice: Optional[int] = None
    useAI: Optional[bool] = True
    numResults: Optional[int] = 5


class AnalogDetail(BaseModel):
    name: str
    avg_price_guess: Optional[int] = None
    note: Optional[str] = None
    pros: list[str] = []
    cons: list[str] = []


class MarketReport(BaseModel):
    item: Optional[str] = None
    market_range: Optional[list[int]] = None
    median_price: Optional[float] = None
    mean_price: Optional[int] = None
    client_price: Optional[int] = None
    client_price_ok: Optional[bool] = None
    explanation: Optional[str] = None


class BestOfferAnalysis(BaseModel):
    best_index: Optional[int] = None
    best_score: Optional[float] = None
    reason: Optional[str] = None
    ranking: list[dict] = []


class BestOffersComparison(BaseModel):
    winner: Optional[str] = None
    original_score: Optional[float] = None
    analog_score: Optional[float] = None
    price_comparison: Optional[dict] = None
    pros_original: list[str] = []
    cons_original: list[str] = []
    pros_analog: list[str] = []
    cons_analog: list[str] = []
    recommendation: Optional[str] = None
    use_cases_original: list[str] = []
    use_cases_analog: list[str] = []
    original_url: Optional[str] = None
    analog_url: Optional[str] = None
    original_title: Optional[str] = None
    analog_title: Optional[str] = None
    comparison_details: Optional[dict] = None
    key_differences: list[str] = []


class DescribeResponse(BaseModel):
    category: Optional[str] = None
    vendor: Optional[str] = None
    model: Optional[str] = None
    price: Optional[int] = None
    currency: Optional[str] = None
    monthly_payment: Optional[int] = None
    year: Optional[int] = None
    condition: Optional[str] = None
    location: Optional[str] = None
    specs: dict = {}
    pros: list[str] = []
    cons: list[str] = []
    analogs_mentioned: list[str] = []

    market_report: MarketReport = MarketReport()
    analogs_details: list[AnalogDetail] = []
    sources: list[dict] = []
    
    # Deep analysis results
    best_original_offer: Optional[dict] = None
    best_original_analysis: Optional[BestOfferAnalysis] = None
    best_offers_comparison: dict[str, BestOffersComparison] = {}


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    index_path = templates_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html не найден</h1>", status_code=404)


@app.post("/api/describe", response_model=DescribeResponse)
async def describe(request: DescribeRequest) -> DescribeResponse:
    """
    Главный эндпоинт API.
    1) Берёт text (склеённое описание) и clientPrice
    2) Запускает run_analysis из parser_b.py
    3) Преобразует результаты в DescribeResponse:
       - market_report: диапазон, медиана, отклонение, комментарий
       - analogs_details: список аналогов для карусели
    """
    item_str = request.text.strip()
    client_price = request.clientPrice
    use_ai = request.useAI if request.useAI is not None else True
    num_results = request.numResults if request.numResults else 5

    print(f"[DEBUG] item={item_str[:80]}...")
    print(f"[DEBUG] client_price={client_price}")
    print(f"[DEBUG] use_ai={use_ai}")
    print(f"[DEBUG] num_results={num_results}")

    try:
        try:
            # Запускаем парсер
            analysis = run_analysis(
                item=item_str,
                client_price=client_price,
                use_ai=use_ai,
                num_results=num_results,
            )
        except OverflowError as e:
            # Защита от int too large to convert to float
            print(f"[WARN] Overflow в run_analysis: {e}")
            analysis = {
                "item": item_str,
                "offers_used": [],
                "analogs_suggested": [],
                "analogs_details": [],
                "market_report": {
                    "item": item_str,
                    "market_range": None,
                    "median_price": None,
                    "mean_price": None,
                    "client_price": client_price,
                    "client_price_ok": None,
                    "explanation": "Не удалось посчитать диапазон: данные цен некорректны.",
                },
            }

        # Извлекаем нужные части
        market_report = analysis.get("market_report") or {}
        offers_used = analysis.get("offers_used") or []
        analogs_details_raw = analysis.get("analogs_details") or []

        # Все объявления для просмотра
        sources_for_response: list[dict] = []
        for o in offers_used:  # все объявления
            sources_for_response.append(
                {
                    "title": o.get("title"),
                    "source": o.get("source"),
                    "url": o.get("url"),
                    "price_str": o.get("price_str"),
                    "price": o.get("price"),
                    "monthly_payment_str": o.get("monthly_payment_str"),
                    "model": o.get("model"),
                    "year": o.get("year"),
                    "condition": o.get("condition"),
                    "location": o.get("location"),
                }
            )


        # Первое предложение (для базовых полей)
        first_offer = offers_used[0] if offers_used else {}

        # Преобразуем аналоги в формат, который ожидает фронт
        analogs_for_response = []
        for analog in analogs_details_raw:
            analogs_for_response.append(
                AnalogDetail(
                    name=analog.get("name", "Аналог"),
                    avg_price_guess=analog.get("avg_price_guess"),
                    note=analog.get("note"),
                    pros=analog.get("pros", []),
                    cons=analog.get("cons", []),
                )
            )

        # Deep analysis results
        best_original_offer = market_report.get("best_original_offer")
        best_original_analysis_raw = market_report.get("best_original_analysis", {})
        best_offers_comparison_raw = market_report.get("best_offers_comparison", {})
        
        # Convert best original analysis
        best_original_analysis = None
        if best_original_analysis_raw:
            best_original_analysis = BestOfferAnalysis(
                best_index=best_original_analysis_raw.get("best_index"),
                best_score=best_original_analysis_raw.get("best_score"),
                reason=best_original_analysis_raw.get("reason"),
                ranking=best_original_analysis_raw.get("ranking", [])
            )
        
        # Convert comparisons
        best_offers_comparison = {}
        for analog_name, comp_data in best_offers_comparison_raw.items():
            best_offers_comparison[analog_name] = BestOffersComparison(
                winner=comp_data.get("winner"),
                original_score=comp_data.get("original_score"),
                analog_score=comp_data.get("analog_score"),
                price_comparison=comp_data.get("price_comparison"),
                pros_original=comp_data.get("pros_original", []),
                cons_original=comp_data.get("cons_original", []),
                pros_analog=comp_data.get("pros_analog", []),
                cons_analog=comp_data.get("cons_analog", []),
                recommendation=comp_data.get("recommendation"),
                use_cases_original=comp_data.get("use_cases_original", []),
                use_cases_analog=comp_data.get("use_cases_analog", []),
                original_url=comp_data.get("original_url"),
                analog_url=comp_data.get("analog_url"),
                original_title=comp_data.get("original_title"),
                analog_title=comp_data.get("analog_title"),
                comparison_details=comp_data.get("comparison_details"),
                key_differences=comp_data.get("key_differences", [])
            )
        
        # Собираем ответ
        return DescribeResponse(
            category=first_offer.get("category"),
            vendor=first_offer.get("vendor"),
            model=first_offer.get("model"),
            price=market_report.get("median_price"),
            currency=first_offer.get("currency", "RUB"),
            monthly_payment=first_offer.get("monthly_payment"),
            year=first_offer.get("year"),
            condition=first_offer.get("condition"),
            location=first_offer.get("location"),
            specs=first_offer.get("specs", {}),
            pros=first_offer.get("pros", []),
            cons=first_offer.get("cons", []),
            analogs_mentioned=analysis.get("analogs_suggested", []),
            market_report=MarketReport(
                item=market_report.get("item"),
                market_range=market_report.get("market_range"),
                median_price=market_report.get("median_price"),
                mean_price=market_report.get("mean_price"),
                client_price=market_report.get("client_price"),
                client_price_ok=market_report.get("client_price_ok"),
                explanation=market_report.get("explanation"),
            ),
            analogs_details=analogs_for_response,
            sources=sources_for_response,
            best_original_offer=best_original_offer,
            best_original_analysis=best_original_analysis,
            best_offers_comparison=best_offers_comparison,
        )

    except Exception as e:
        print(f"[ERROR] /api/describe failed: {e}")
        import traceback
        traceback.print_exc()
        
        return DescribeResponse(
            category="Ошибка анализа",
            vendor=str(e)[:100],
            specs={},
            pros=[],
            cons=[],
            analogs_mentioned=[],
            market_report=MarketReport(
                explanation=f"Ошибка: {str(e)[:200]}"
            ),
            analogs_details=[],
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
