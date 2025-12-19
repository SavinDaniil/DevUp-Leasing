const form = document.getElementById("analyzeForm");
const error = document.getElementById("error");
const placeholder = document.getElementById("placeholder");
const loading = document.getElementById("loading");
const resultContent = document.getElementById("resultContent");
const prevBtn = document.querySelector(".slider-btn.prev");
const nextBtn = document.querySelector(".slider-btn.next");

let currentAnalogIndex = 0;
let analogsData = [];
let loadingInterval = null;

// ✅ ДОБАВЛЕНЫ: Переменные для управления запросом
let abortController = null;
let timeoutId = null;

// ===== ПЕРЕКЛЮЧАТЕЛЬ ИИ =====
document.querySelectorAll(".ai-btn").forEach((btn) => {
  btn.addEventListener("click", (e) => {
    e.preventDefault();
    document.querySelectorAll(".ai-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("useAI").value = btn.dataset.value;
    console.log("[DEBUG] useAI =", btn.dataset.value);
  });
});

// ===== АНИМАЦИЯ ЗАГРУЗКИ =====
function startLoadingAnimation() {
  const steps = ["step1", "step2", "step3", "step4"];
  let currentStep = 0;

  steps.forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.classList.remove("active", "done");
    }
  });

  const firstStep = document.getElementById(steps[0]);
  if (firstStep) firstStep.classList.add("active");

  loadingInterval = setInterval(() => {
    const currentEl = document.getElementById(steps[currentStep]);
    if (currentEl) {
      currentEl.classList.remove("active");
      currentEl.classList.add("done");
    }

    currentStep++;
    if (currentStep < steps.length) {
      const nextEl = document.getElementById(steps[currentStep]);
      if (nextEl) nextEl.classList.add("active");
    } else {
      currentStep = 0;
      steps.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.remove("done");
      });
      const firstEl = document.getElementById(steps[0]);
      if (firstEl) firstEl.classList.add("active");
    }
  }, 2500);
}

function stopLoadingAnimation() {
  if (loadingInterval) {
    clearInterval(loadingInterval);
    loadingInterval = null;
  }
}

// ===== ВАЛИДАЦИЯ ФОРМЫ =====
function validateForm() {
  const item = document.getElementById("item").value.trim();
  const clientPrice = document.getElementById("clientPrice").value.trim();
  const numResults = parseInt(document.getElementById("numResults").value, 10);

  if (!item || item.length < 3) {
    error.textContent = "❌ Описание должно содержать минимум 3 символа";
    error.classList.add("show");
    return false;
  }

  if (item.length > 500) {
    error.textContent = "❌ Описание не должно превышать 500 символов";
    error.classList.add("show");
    return false;
  }

  if (clientPrice) {
    const price = parseInt(clientPrice, 10);
    if (isNaN(price) || price < 0 || price > 10**12) {
      error.textContent = "❌ Цена должна быть числом от 0 до 1 триллиона";
      error.classList.add("show");
      return false;
    }
  }

  if (isNaN(numResults) || numResults < 1 || numResults > 10) {
    error.textContent = "❌ Количество результатов должно быть от 1 до 10";
    error.classList.add("show");
    return false;
  }

  return true;
}

// ===== ОТПРАВКА ФОРМЫ =====
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  error.classList.remove("show");

  if (!validateForm()) {
    return;
  }

  form.querySelector("button").disabled = true;
  console.log("[DEBUG] Форма отправлена");

  placeholder.classList.add("hidden");
  resultContent.classList.remove("show");
  loading.classList.add("show");

  // ✅ ИСПРАВЛЕНО: Запуск анимации загрузки
  startLoadingAnimation();

  const item = document.getElementById("item").value.trim();
  const clientPrice = parseInt(document.getElementById("clientPrice").value, 10) || null;
  const useAI = document.getElementById("useAI").value === "true";
  const numResults = parseInt(document.getElementById("numResults").value, 10) || 5;

  console.log("[DEBUG] item:", item);
  console.log("[DEBUG] clientPrice:", clientPrice);
  console.log("[DEBUG] useAI:", useAI);
  console.log("[DEBUG] numResults:", numResults);

  try {
    // ✅ ИСПРАВЛЕНО: Создание новых controller и timeout для каждого запроса
    abortController = new AbortController();
    
    // Таймаут увеличен до 20 минут (1200000 мс) для долгих анализов
    timeoutId = setTimeout(() => {
      abortController.abort();
      stopLoadingAnimation();
      loading.classList.remove("show");
      error.textContent = "⏱️ Время ожидания истекло (20 минут). Анализ занял слишком много времени. Попробуйте упростить запрос или уменьшить количество результатов.";
      error.classList.add("show");
    }, 1200000); // 20 минут

    const resp = await fetch("/api/describe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: item, clientPrice, useAI, numResults }),
      signal: abortController.signal,
    });

    clearTimeout(timeoutId);

    if (!resp.ok) {
      // ✅ ИСПРАВЛЕНО: Показать ошибку сервера
      const errorData = await resp.json().catch(() => ({}));
      throw new Error(errorData.detail || `Ошибка: ${resp.status} ${resp.statusText}`);
    }

    const data = await resp.json();
    console.log("[DEBUG] Данные распарсены:", JSON.stringify(data, null, 2));

    // ✅ Задержка для эффекта
    await new Promise((resolve) => setTimeout(resolve, 1200));

    analogsData = data.analogs_details || [];
    currentAnalogIndex = 0;

    render(data, clientPrice);
    renderSources(data.sources || []);
    renderBestOriginal(data);
    renderBestComparison(data);
    renderAllOffers(data.sources || []);

    // ✅ ИСПРАВЛЕНО: Остановка анимации загрузки
    stopLoadingAnimation();
    loading.classList.remove("show");
    resultContent.classList.add("show");

    if (analogsData.length > 0) {
      showAnalog(0);
    } else {
      document.getElementById("analogCard").innerHTML = "<p style='color: var(--muted)'>Аналоги не найдены</p>";
      prevBtn.disabled = true;
      nextBtn.disabled = true;
      updateAnalogCounter();
    }
  } catch (err) {
    // ✅ ИСПРАВЛЕНО: Обработка ошибок с информативным сообщением
    stopLoadingAnimation();
    loading.classList.remove("show");
    
    console.error("[ERROR] Ошибка при анализе:", err);
    
    if (err.name === "AbortError") {
      // Проверяем, был ли это таймаут или отмена пользователем
      if (timeoutId) {
        error.textContent = "⏱️ Время ожидания истекло (20 минут). Анализ занял слишком много времени. Попробуйте упростить запрос или уменьшить количество результатов.";
      } else {
        error.textContent = "❌ Запрос был отменен. Проверьте соединение с сервером.";
      }
    } else if (err.message) {
      // Показываем сообщение об ошибке от сервера
      error.textContent = `❌ ${err.message}`;
    } else {
      error.textContent = "❌ Произошла неизвестная ошибка. Попробуйте позже или проверьте соединение.";
    }
    error.classList.add("show");
  } finally {
    clearTimeout(timeoutId);
    timeoutId = null;
    abortController = null;
    form.querySelector("button").disabled = false;
  }
});

// ===== РЕНДЕР МЕТРИК =====
function render(data, clientPrice) {
  console.log("[DEBUG] Рендеринг данных:", data);

  const titleEl = document.getElementById("resultTitle");
  if (titleEl) {
    const itemName = data.vendor && data.model
      ? `${data.vendor} ${data.model}`
      : data.market_report?.item || "Результат анализа";
    titleEl.textContent = `📊 Анализ: ${itemName}`;
  }

  const marketReport = data.market_report || {};
  const minPrice = marketReport.market_range ? marketReport.market_range[0] : null;
  const maxPrice = marketReport.market_range ? marketReport.market_range[1] : null;
  const medianPrice = marketReport.median_price;

  const formatPrice = (price) => {
    if (!price) return "—";
    return `${price.toLocaleString("ru-RU")} ₽`;
  };

  if (minPrice && maxPrice) {
    document.getElementById("rangeValue").textContent = `${formatPrice(minPrice)} – ${formatPrice(maxPrice)}`;
  } else {
    document.getElementById("rangeValue").textContent = "—";
  }

  document.getElementById("medianValue").textContent = formatPrice(medianPrice);
  document.getElementById("clientValue").textContent = formatPrice(clientPrice);

  if (clientPrice && medianPrice) {
    const deviation = Math.round(((clientPrice - medianPrice) / medianPrice) * 100);
    const deviationText = deviation > 0 ? `+${deviation}%` : `${deviation}%`;
    const color = Math.abs(deviation) <= 20 ? "green" : "red";
    document.getElementById("deviationValue").innerHTML = `<span style="color: ${color};">${deviationText}</span>`;
  } else {
    document.getElementById("deviationValue").textContent = "—";
  }

  let comment = `<strong>${data.vendor || "—"} ${data.model || "—"}</strong>`;
  if (data.category) comment += `<br>${data.category}`;
  if (data.year) comment += `<br>${data.year}`;
  if (data.condition) comment += `<br>${data.condition}`;
  if (marketReport.explanation) comment += `<br><br><strong>Объяснение рынка:</strong><br>${marketReport.explanation}`;

  document.getElementById("commentSection").innerHTML = comment;
}

// ===== ПОКАЗАТЬ АНАЛОГ =====
function showAnalog(index) {
  if (analogsData.length === 0) {
    document.getElementById("analogCard").innerHTML = "<p style='color: var(--muted)'>Аналоги не найдены</p>";
    prevBtn.disabled = true;
    nextBtn.disabled = true;
    updateAnalogCounter();
    return;
  }

  currentAnalogIndex = Math.max(0, Math.min(index, analogsData.length - 1));
  const analog = analogsData[currentAnalogIndex];

  let html = `<div class="analog-name">${analog.name}</div>`;

  if (analog.avg_price_guess) {
    const price = analog.avg_price_guess.toLocaleString("ru-RU");
    html += `<div class="analog-price">${price} ₽</div>`;
  }

  if (analog.note) {
    html += `<div class="analog-note">${analog.note}</div>`;
  }

  if (analog.pros && analog.pros.length > 0) {
    html += `<div class="analog-pros">
      <div style="color: #4ade80; font-size: 11px; font-weight: 600; margin-bottom: 6px;">✓ Преимущества</div>
      <ul class="analog-list">`;
    analog.pros.forEach(p => {
      html += `<li>${p}</li>`;
    });
    html += `</ul></div>`;
  }

  if (analog.cons && analog.cons.length > 0) {
    html += `<div class="analog-cons">
      <div style="color: #fb7185; font-size: 11px; font-weight: 600; margin-bottom: 6px;">✗ Недостатки</div>
      <ul class="analog-list">`;
    analog.cons.forEach(c => {
      html += `<li>${c}</li>`;
    });
    html += `</ul></div>`;
  }

  document.getElementById("analogCard").innerHTML = html;
  updateAnalogCounter();
  prevBtn.disabled = currentAnalogIndex === 0;
  nextBtn.disabled = currentAnalogIndex === analogsData.length - 1;
}

function nextAnalog() {
  showAnalog(currentAnalogIndex + 1);
}

function prevAnalog() {
  showAnalog(currentAnalogIndex - 1);
}

function updateAnalogCounter() {
  const total = analogsData.length;
  const current = analogsData.length > 0 ? currentAnalogIndex + 1 : 0;
  document.getElementById("analogCounter").textContent = `${current}/${total}`;
}

prevBtn.addEventListener("click", prevAnalog);
nextBtn.addEventListener("click", nextAnalog);

// ===== РЕНДЕР ИСТОЧНИКОВ =====
function renderSources(sources) {
  const list = document.getElementById("sourcesList");
  if (!list) return;

  list.innerHTML = "";

  if (!sources || sources.length === 0) {
    list.innerHTML = "<li style='color: var(--muted); font-size: 12px;'>Источники не найдены</li>";
    return;
  }

  sources.forEach(s => {
    const li = document.createElement("li");
    const title = s.title || "Без названия";
    const src = s.source ? s.source : "";
    const price = s.pricestr ? s.pricestr : "";

    // Всегда показываем заголовок, ссылка опциональна
    if (s.url) {
      li.innerHTML = `<a href="${s.url}" target="_blank" rel="noopener noreferrer">${title}</a><span style="color: var(--muted); font-size: 11px;">${src} ${price}</span>`;
    } else {
      li.innerHTML = `<span>${title}</span><span style="color: var(--muted); font-size: 11px;">${src} ${price}</span>`;
    }

    list.appendChild(li);
  });
}

// ===== РЕНДЕР ЛУЧШЕГО ОРИГИНАЛЬНОГО ПРЕДЛОЖЕНИЯ =====
function renderBestOriginal(data) {
  const section = document.getElementById("bestOriginalSection");
  const card = document.getElementById("bestOriginalCard");
  if (!section || !card) return;

  const bestOffer = data.best_original_offer;
  const analysis = data.best_original_analysis;

  if (!bestOffer || !analysis) {
    section.classList.add("hidden");
    return;
  }

  section.classList.remove("hidden");

  let html = `<div class="best-offer-title">${bestOffer.title}</div>`;

  if (bestOffer.url) {
    html += `<div class="best-offer-url"><a href="${bestOffer.url}" target="_blank">${bestOffer.url}</a></div>`;
  }

  if (bestOffer.pricestr) {
    html += `<div style="font-size: 14px; margin: 8px 0;"><strong>${bestOffer.pricestr}</strong></div>`;
  }

  if (bestOffer.year) {
    html += `<div style="font-size: 13px; color: var(--muted);">${bestOffer.year}</div>`;
  }

  if (bestOffer.condition) {
    html += `<div style="font-size: 13px; color: var(--muted);">${bestOffer.condition}</div>`;
  }

  const score = analysis.best_score || 0;
  html += `<div class="best-offer-score">⭐ ${score.toFixed(1)}/10</div>`;

  if (analysis.reason) {
    html += `<div class="best-offer-reason">${analysis.reason}</div>`;
  }

  card.innerHTML = html;
}

// ===== РЕНДЕР СРАВНЕНИЯ ЛУЧШИХ ПРЕДЛОЖЕНИЙ =====
function renderBestComparison(data) {
  const section = document.getElementById("bestComparisonSection");
  const content = document.getElementById("bestComparisonContent");
  if (!section || !content) return;

  const comparisons = data.best_offers_comparison;

  if (Object.keys(comparisons).length === 0) {
    section.classList.add("hidden");
    return;
  }

  section.classList.remove("hidden");
  content.innerHTML = "";

  for (const [analogName, comp] of Object.entries(comparisons)) {
    const div = document.createElement("div");
    div.className = "comparison-item";

    let html = `<div class="comparison-header">
      <strong>Оригинал</strong> vs <strong>${analogName}</strong>
    </div>`;

    html += `<div class="comparison-winner">🏆 Лучший выбор: <strong>${comp.winner === "original" ? "Оригинал" : analogName}</strong></div>`;

    html += `<div class="comparison-scores">
      <div class="comparison-score">Оригинал: ${comp.original_score || 0.0.toFixed(1)}/10</div>
      <div class="comparison-score">Аналог: ${comp.analog_score || 0.0.toFixed(1)}/10</div>
    </div>`;

    // Ссылки на объявления
    html += `<div class="comparison-links">`;
    if (comp.original_url) {
      html += `<div class="offer-link">
        <strong>Оригинал:</strong>
        <a href="${comp.original_url}" target="_blank" rel="noopener noreferrer">${comp.original_title || comp.original_url}</a>
      </div>`;
    }
    if (comp.analog_url) {
      html += `<div class="offer-link">
        <strong>Аналог:</strong>
        <a href="${comp.analog_url}" target="_blank" rel="noopener noreferrer">${comp.analog_title || comp.analog_url}</a>
      </div>`;
    }
    html += `</div>`;

    // Детальное сравнение
    if (comp.comparison_details) {
      html += `<div class="comparison-details">
        <h4>Детали сравнения</h4>`;
      if (comp.comparison_details.price) {
        html += `<div class="detail-item"><strong>Цена:</strong> ${comp.comparison_details.price}</div>`;
      }
      if (comp.comparison_details.quality) {
        html += `<div class="detail-item"><strong>Качество:</strong> ${comp.comparison_details.quality}</div>`;
      }
      if (comp.comparison_details.financing) {
        html += `<div class="detail-item"><strong>Финансирование:</strong> ${comp.comparison_details.financing}</div>`;
      }
      if (comp.comparison_details.reliability) {
        html += `<div class="detail-item"><strong>Надежность:</strong> ${comp.comparison_details.reliability}</div>`;
      }
      if (comp.comparison_details.value) {
        html += `<div class="detail-item"><strong>Ценность:</strong> ${comp.comparison_details.value}</div>`;
      }
      html += `</div>`;
    }

    // Ключевые различия
    if (comp.key_differences && comp.key_differences.length > 0) {
      html += `<div class="key-differences">
        <h4>Ключевые различия</h4>
        <ul>`;
      comp.key_differences.forEach(diff => {
        html += `<li>${diff}</li>`;
      });
      html += `</ul></div>`;
    }

    // Сравнение цен
    if (comp.price_comparison) {
      const pc = comp.price_comparison;
      const origPrice = pc.original_price ? pc.original_price.toLocaleString("ru-RU") : "—";
      const analogPrice = pc.analog_price ? pc.analog_price.toLocaleString("ru-RU") : "—";
      const diff = pc.difference_percent ? pc.difference_percent.toFixed(1) : 0;

      html += `<div class="comparison-price">
        <strong>Оригинал:</strong> ${origPrice} ₽`;
      if (pc.monthly_payment_original) {
        html += ` (${pc.monthly_payment_original.toLocaleString("ru-RU")} ₽/мес)`;
      }
      html += `<br><strong>Аналог:</strong> ${analogPrice} ₽`;
      if (pc.monthly_payment_analog) {
        html += ` (${pc.monthly_payment_analog.toLocaleString("ru-RU")} ₽/мес)`;
      }
      if (diff !== 0) {
        html += `<br><strong>Разница: ${diff > 0 ? '+' : ''}${diff}%</strong>`;
      }
      html += `</div>`;
    }

    // Плюсы и минусы
    html += `<div class="comparison-pros-cons">`;

    if (comp.pros_original && comp.pros_original.length > 0) {
      html += `<div class="comparison-pros">
        <h4 style="color: var(--accent);">✓ Плюсы оригинала</h4>
        <ul>`;
      comp.pros_original.slice(0, 3).forEach(p => {
        html += `<li>${p}</li>`;
      });
      html += `</ul></div>`;
    }

    if (comp.cons_original && comp.cons_original.length > 0) {
      html += `<div class="comparison-cons">
        <h4 style="color: var(--danger);">✗ Минусы оригинала</h4>
        <ul>`;
      comp.cons_original.slice(0, 3).forEach(c => {
        html += `<li>- ${c}</li>`;
      });
      html += `</ul></div>`;
    }

    if (comp.pros_analog && comp.pros_analog.length > 0) {
      html += `<div class="comparison-pros">
        <h4 style="color: var(--accent);">✓ Плюсы аналога</h4>
        <ul>`;
      comp.pros_analog.slice(0, 3).forEach(p => {
        html += `<li>${p}</li>`;
      });
      html += `</ul></div>`;
    }

    if (comp.cons_analog && comp.cons_analog.length > 0) {
      html += `<div class="comparison-cons">
        <h4 style="color: var(--danger);">✗ Минусы аналога</h4>
        <ul>`;
      comp.cons_analog.slice(0, 3).forEach(c => {
        html += `<li>- ${c}</li>`;
      });
      html += `</ul></div>`;
    }

    html += `</div>`;

    // Рекомендация
    if (comp.recommendation) {
      html += `<div class="comparison-recommendation">
        <strong>Рекомендация:</strong><br>${comp.recommendation}
      </div>`;
    }

    div.innerHTML = html;
    content.appendChild(div);
  }
}

// ===== РЕНДЕР ВСЕХ ОБЪЯВЛЕНИЙ =====
function renderAllOffers(sources) {
  const toggleBtn = document.getElementById("toggleAllOffers");
  const section = document.getElementById("allOffersSection");
  const list = document.getElementById("allOffersList");

  if (!toggleBtn || !section || !list) return;

  if (!sources || sources.length === 0) {
    toggleBtn.style.display = "none";
    return;
  }

  toggleBtn.style.display = "block";
  list.innerHTML = "";

  sources.forEach((offer, index) => {
    const div = document.createElement("div");
    div.className = "offer-item";

    let html = `<div class="offer-item-header">
      <span class="offer-number">${index + 1}</span>`;

    if (offer.url) {
      html += `<a href="${offer.url}" target="_blank" rel="noopener noreferrer" class="offer-title-link">${offer.title}</a>`;
    } else {
      html += `<span class="offer-title">${offer.title}</span>`;
    }

    html += `</div>`;

    html += `<div class="offer-item-details">`;
    if (offer.source) html += `<span class="offer-source">${offer.source}</span>`;
    if (offer.pricestr) html += `<span class="offer-price">${offer.pricestr}</span>`;
    if (offer.monthly_payment_str) html += `<span class="offer-payment">${offer.monthly_payment_str}</span>`;
    if (offer.year) html += `<span class="offer-year">${offer.year}</span>`;
    if (offer.condition) html += `<span class="offer-condition">${offer.condition}</span>`;
    if (offer.location) html += `<span class="offer-location">${offer.location}</span>`;
    html += `</div>`;

    div.innerHTML = html;
    list.appendChild(div);
  });

  const titleEl = document.getElementById("allOffersTitle");
  if (titleEl) {
    titleEl.textContent = `Все объявления (${sources.length})`;
  }
}

// ===== ОБРАБОТЧИК КНОПКИ ПЕРЕКЛЮЧЕНИЯ =====
const toggleBtn = document.getElementById("toggleAllOffers");
if (toggleBtn) {
  toggleBtn.onclick = () => {
    const section = document.getElementById("allOffersSection");
    const isHidden = section.classList.contains("hidden");

    if (isHidden) {
      section.classList.remove("hidden");
      toggleBtn.textContent = "Скрыть все объявления";
    } else {
      section.classList.add("hidden");
      toggleBtn.textContent = `Показать все объявления (${document.getElementById("allOffersList").children.length})`;
    }
  };
}
