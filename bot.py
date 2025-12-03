import os
import logging
from typing import Dict, Any

import requests
from dotenv import load_dotenv
from telegram import (
    Update,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from telegram.ext import (
    Application,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ---------- ЛОГИ ----------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------- ЗАГРУЗКА КОНФИГА ----------

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/analyze")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN не найден в .env")

# ---------- СОСТОЯНИЯ ДИАЛОГА ----------

TYPE, MODEL, YEAR, PARAMS, PRICE, CONFIRM = range(6)

# ---------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------


def _cleanup_price(raw: str) -> float:
    """
    Преобразует строку с ценой в float.
    Примеры: '10 500 000', '10,5' -> 10500000.0 / 10.5
    """
    cleaned = raw.replace(" ", "").replace(",", ".")
    return float(cleaned)


def _build_result_text(result: Dict[str, Any]) -> str:
    """
    Формирует markdown‑текст отчёта из ответа backend.
    Ожидается структура:
    {
      "median_price": 9500000,
      "price_deviation_pct": 5.3,
      "decision": "confirmed",
      "comment": "...",
      "analogs": [
        {
          "title": "...",
          "price": 9300000,
          "params": "...",
          "source_name": "Avito",
          "url": "https://..."
        }
      ]
    }
    """
    analogs = result.get("analogs", [])
    median_price = result.get("median_price")
    deviation = result.get("price_deviation_pct")
    decision = result.get("decision", "")
    comment = result.get("comment", "")

    lines = []

    # Таблица аналогов
    if analogs:
        lines.append("## Аналоги на рынке\n")
        lines.append("| Аналог | Цена, млн руб | Параметры | Источник |")
        lines.append("|--------|----------------|-----------|----------|")
        for a in analogs:
            title = a.get("title", "-")
            price_mln = (a.get("price") or 0) / 1_000_000
            params = a.get("params", "-")
            source_name = a.get("source_name", "-")
            lines.append(
                f"| {title} | {price_mln:.2f} | {params} | {source_name} |"
            )
        lines.append("")  # пустая строка после таблицы

    # Блок с цифрами
    if median_price is not None:
        lines.append(
            f"Медианная рыночная цена: {median_price:,.0f} руб.".replace(",", " ")
        )
    if deviation is not None:
        lines.append(
            f"Отклонение заявленной цены: {deviation:+.1f}%."
        )
    if decision:
        lines.append(f"**Вывод:** {decision}.")
    if comment:
        lines.append(comment)

    return "\n".join(lines)


# ---------- ОБРАБОТЧИКИ КОМАНД ----------


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /start — приветствие и краткое объяснение.
    """
    text = (
        "Привет! Это бот для оценки рыночной стоимости предмета лизинга.\n\n"
        "Нажми /analyze, чтобы начать анализ нового объекта."
    )
    await update.message.reply_text(text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /help — краткая подсказка.
    """
    text = (
        "Доступные команды:\n"
        "/start — начать работу\n"
        "/analyze — запустить анализ предмета лизинга\n"
        "/cancel — отменить текущий диалог"
    )
    await update.message.reply_text(text)


# ---------- ДИАЛОГ: ШАГ 1 — СТАРТ АНАЛИЗА ----------


async def analyze_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    /analyze — запуск сценария, спрашиваем тип предмета.
    """
    reply_keyboard = [
        ["Транспорт", "Спецтехника"],
        ["Оборудование", "Другое"],
    ]
    await update.message.reply_text(
        "Выберите тип предмета лизинга:",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True, resize_keyboard=True
        ),
    )
    return TYPE


# ---------- ДИАЛОГ: ШАГ 2 — ТИП ----------


async def handle_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Сохраняем тип, спрашиваем марку/модель.
    """
    item_type = update.message.text.strip()
    context.user_data["item_type"] = item_type

    await update.message.reply_text(
        "Введите марку и модель (например, MAN TGS 26.440 или "
        "«Буровая установка XYZ‑123»):",
        reply_markup=ReplyKeyboardRemove(),
    )
    return MODEL


# ---------- ДИАЛОГ: ШАГ 3 — МОДЕЛЬ ----------


async def handle_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Сохраняем марку/модель, спрашиваем год выпуска.
    """
    model = update.message.text.strip()
    context.user_data["brand_model"] = model

    await update.message.reply_text("Введите год выпуска (например, 2019):")
    return YEAR


# ---------- ДИАЛОГ: ШАГ 4 — ГОД ----------


async def handle_year(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Сохраняем год выпуска, проверяем, что это число, спрашиваем характеристики.
    """
    text = update.message.text.strip()
    try:
        year = int(text)
    except ValueError:
        await update.message.reply_text(
            "Пожалуйста, введите год числом, например: 2019."
        )
        return YEAR

    context.user_data["year"] = year

    await update.message.reply_text(
        "Опишите ключевые характеристики (пробег/наработка, мощность, "
        "грузоподъёмность и т.д.):"
    )
    return PARAMS


# ---------- ДИАЛОГ: ШАГ 5 — ХАРАКТЕРИСТИКИ ----------


async def handle_params(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Сохраняем характеристики, спрашиваем заявленную стоимость.
    """
    params = update.message.text.strip()
    context.user_data["params"] = params

    await update.message.reply_text(
        "Укажите заявленную стоимость (в рублях, можно с пробелами):"
    )
    return PRICE


# ---------- ДИАЛОГ: ШАГ 6 — ЦЕНА ----------


async def handle_price(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Сохраняем цену, показываем резюме и спрашиваем подтверждение.
    """
    raw_price = update.message.text.strip()

    try:
        price = _cleanup_price(raw_price)
    except ValueError:
        await update.message.reply_text(
            "Не получилось распознать число. Введите стоимость ещё раз, "
            "например: 10500000 или 10 500 000."
        )
        return PRICE

    context.user_data["declared_price"] = price

    # Формируем резюме введённых данных
    item_type = context.user_data.get("item_type")
    brand_model = context.user_data.get("brand_model")
    year = context.user_data.get("year")
    params = context.user_data.get("params")

    summary = (
        "Проверьте, всё ли верно:\n\n"
        f"Тип: {item_type}\n"
        f"Модель: {brand_model}\n"
        f"Год выпуска: {year}\n"
        f"Характеристики: {params}\n"
        f"Заявленная стоимость: {price:,.0f} руб.\n\n"
        "Запустить анализ?"
    ).replace(",", " ")

    reply_keyboard = [["Да", "Отмена"]]

    await update.message.reply_text(
        summary,
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True, resize_keyboard=True
        ),
    )
    return CONFIRM


# ---------- ДИАЛОГ: ШАГ 7 — ПОДТВЕРЖДЕНИЕ И ВЫЗОВ BACKEND ----------


async def handle_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Если пользователь подтверждает, шлём данные на backend и показываем результат.
    """
    choice = update.message.text.strip().lower()

    if choice.startswith("отмена") or choice == "/cancel":
        await update.message.reply_text(
            "Анализ отменён.", reply_markup=ReplyKeyboardRemove()
        )
        return ConversationHandler.END

    if not choice.startswith("да"):
        await update.message.reply_text(
            "Пожалуйста, выберите 'Да' для запуска анализа или 'Отмена' "
            "для выхода."
        )
        return CONFIRM

    # Сообщение о начале анализа
    await update.message.reply_text(
        "Запускаю поиск аналогов и анализ рынка, это может занять до минуты...",
        reply_markup=ReplyKeyboardRemove(),
    )

    # Собираем данные для backend
    payload = {
        "item_type": context.user_data.get("item_type"),
        "brand_model": context.user_data.get("brand_model"),
        "year": context.user_data.get("year"),
        "params": context.user_data.get("params"),
        "declared_price": context.user_data.get("declared_price"),
    }

    logger.info("Отправка payload в backend: %s", payload)

    try:
        resp = requests.post(BACKEND_URL, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        logger.exception("Ошибка при запросе к backend: %s", e)
        await update.message.reply_text(
            "Не удалось получить ответ от сервиса оценки. "
            "Попробуйте позже или обратитесь к ответственному за систему."
        )
        return ConversationHandler.END

    # Формируем текст для пользователя
    text = _build_result_text(result)
    if not text:
        text = "Сервис вернул пустой результат. Попробуйте позже."

    await update.message.reply_markdown(text)
    return ConversationHandler.END


# ---------- ОТМЕНА ДИАЛОГА ----------


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    /cancel — универсальная отмена диалога.
    """
    await update.message.reply_text(
        "Диалог прерван. Можно начать заново командой /analyze.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


# ---------- ТОЧКА ВХОДА ----------


def main() -> None:
    """
    Создаёт приложение и запускает long polling.
    """
    application = Application.builder().token(BOT_TOKEN).build()

    # Обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # Конфигурация многошагового диалога
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("analyze", analyze_start)],
        states={
            TYPE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_type)
            ],
            MODEL: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_model)
            ],
            YEAR: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_year)
            ],
            PARAMS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_params)
            ],
            PRICE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_price)
            ],
            CONFIRM: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_confirm)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler)

    logger.info("Бот запущен. Ожидаем сообщения...")
    application.run_polling()


if __name__ == "__main__":
    main()
