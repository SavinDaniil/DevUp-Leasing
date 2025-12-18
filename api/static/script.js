const form = document.getElementById("analyzeForm");
const error = document.getElementById("error");
const placeholder = document.getElementById("placeholder");
const loading = document.getElementById("loading");
const resultContent = document.getElementById("resultContent");
const prevBtn = document.querySelector(".slider-btn.prev");
const nextBtn = document.querySelector(".slider-btn.next");

let currentAnalogIndex = 0;
let analogsData = [];

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

// ===== ОТПРАВКА ФОРМЫ =====
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  error.classList.remove("show");
  form.querySelector("button").disabled = true;

  console.log("[DEBUG] Форма отправлена");

  // UI состояния
  placeholder.classList.add("hidden");
  resultContent.classList.remove("show");
  loading.classList.add("show");

  const item = document.getElementById("item").value.trim();
  const clientPrice = parseInt(document.getElementById("clientPrice").value, 10) || null;
  const useAI = document.getElementById("useAI").value === "true";
  const numResults = parseInt(document.getElementById("numResults").value, 10) || 5;

  console.log("[DEBUG] item:", item);
  console.log("[DEBUG] clientPrice:", clientPrice);
  console.log("[DEBUG] useAI:", useAI);
  console.log("[DEBUG] numResults:", numResults);

  try {
    console.log("[DEBUG] Отправляем запрос на /api/describe");

    const resp = await fetch("/api/describe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: item,
        clientPrice,
        useAI,
        numResults,
      }),
    });

    console.log("[DEBUG] Ответ получен, статус:", resp.status);

    if (!resp.ok) {
      const errText = await resp.text();
      console.error("[ERROR] Ответ ошибки:", errText);
      throw new Error("Ошибка сервера: " + resp.status);
    }

     const data = await resp.json();
    console.log("[DEBUG] Данные распарсены:", JSON.stringify(data, null, 2));

    // Задержка для эффекта
    await new Promise((resolve) => setTimeout(resolve, 1200));

    analogsData = data.analogs_details || [];
    currentAnalogIndex = 0;

    render(data, clientPrice);
    renderSources(data.sources || []);
    renderBestOriginal(data);
    renderBestComparison(data);
    renderAllOffers(data.sources || []);

    loading.classList.remove("show");
    resultContent.classList.add("show");

    if (analogsData.length > 0) {
      showAnalog(0);
    } else {
      document.getElementById("analogCard").innerHTML =
        "<p style='color: var(--muted);'>Аналоги не найдены</p>";
      prevBtn.disabled = true;
      nextBtn.disabled = true;
      updateAnalogCounter();
    }
  } catch (err) {
    console.error("[ERROR]", err.message);
    loading.classList.remove("show");
    placeholder.classList.remove("hidden");
    error.classList.add("show");
    error.textContent = "Ошибка: " + err.message;
  } finally {
    form.querySelector("button").disabled = false;
  }
});

// ===== РЕНДЕР МЕТРИК =====
function render(data, clientPrice) {
  console.log("[DEBUG] Рендеринг данных:", data);

  // Update title
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
    document.getElementById("rangeValue").textContent =
      `${formatPrice(minPrice)} – ${formatPrice(maxPrice)}`;
  } else {
    document.getElementById("rangeValue").textContent = "—";
  }

  document.getElementById("medianValue").textContent = formatPrice(medianPrice);
  document.getElementById("clientValue").textContent = formatPrice(clientPrice);

  if (clientPrice && medianPrice) {
    const deviation = Math.round(((clientPrice - medianPrice) / medianPrice) * 100);
    const deviationText = deviation > 0 ? `+${deviation}%` : `${deviation}%`;
    const color = Math.abs(deviation) <= 20 ? "green" : "red";
    document.getElementById("deviationValue").innerHTML =
      `<span style="color: ${color};">${deviationText}</span>`;
  } else {
    document.getElementById("deviationValue").textContent = "—";
  }

  let comment = `<strong>${data.vendor || ""} ${data.model || ""}</strong>`;
  if (data.category) comment += `<br>📂 Категория: ${data.category}`;
  if (data.year) comment += `<br>📅 Год: ${data.year}`;
  if (data.condition) comment += `<br>🔧 Состояние: ${data.condition}`;

  if (marketReport.explanation) {
    comment += `<br><br><strong>Рыночная оценка:</strong><br>${marketReport.explanation}`;
  }

  document.getElementById("commentSection").innerHTML = comment;
}

// ===== СЛАЙДЕР АНАЛОГОВ =====
function showAnalog(index) {
  if (analogsData.length === 0) {
    document.getElementById("analogCard").innerHTML =
      "<p style='color: var(--muted);'>Аналоги не найдены</p>";
    prevBtn.disabled = true;
    nextBtn.disabled = true;
    updateAnalogCounter();
    return;
  }

  currentAnalogIndex = Math.max(0, Math.min(index, analogsData.length - 1));
  const analog = analogsData[currentAnalogIndex];

  let html = `<div class="analog-name">${analog.name || "Аналог"}</div>`;

  if (analog.avg_price_guess) {
    const price = analog.avg_price_guess.toLocaleString("ru-RU");
    html += `<div class="analog-price">~${price} ₽</div>`;
  }

  if (analog.note) {
    html += `<div class="analog-note">${analog.note}</div>`;
  }

  if (analog.pros && analog.pros.length > 0) {
    html += '<div class="analog-pros">';
    html += '<div style="color: #4ade80; font-size: 11px; font-weight: 600; margin-bottom: 6px;">✓ ПЛЮСЫ</div>';
    html += '<ul class="analog-list">';
    analog.pros.forEach((p) => {
      html += `<li>${p}</li>`;
    });
    html += "</ul></div>";
  }

  if (analog.cons && analog.cons.length > 0) {
    html += '<div class="analog-cons">';
    html += '<div style="color: #fb7185; font-size: 11px; font-weight: 600; margin-bottom: 6px;">✗ МИНУСЫ</div>';
    html += '<ul class="analog-list">';
    analog.cons.forEach((c) => {
      html += `<li>${c}</li>`;
    });
    html += "</ul></div>";
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

function renderSources(sources) {
  const list = document.getElementById("sourcesList");
  if (!list) return;

  list.innerHTML = "";

  if (!sources || sources.length === 0) {
    list.innerHTML =
      "<li style='color: var(--muted); font-size: 12px;'>Источники не найдены</li>";
    return;
  }

  sources.forEach((s) => {
    const li = document.createElement("li");
    const title = s.title || "Объявление";
    const src = s.source ? ` (${s.source})` : "";
    const price = s.price_str ? ` · ${s.price_str}` : "";

    if (s.url) {
      li.innerHTML = `<a href="${s.url}" target="_blank" rel="noopener noreferrer">${title}</a>${src}${price}`;
    } else {
      li.textContent = `${title}${src}${price}`;
    }
    list.appendChild(li);
  });
}

// ===== RENDER BEST ORIGINAL OFFER =====
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
  
  let html = `<div class="best-offer-title">${bestOffer.title || "Лучшее объявление"}</div>`;
  
  if (bestOffer.url) {
    html += `<div class="best-offer-url"><a href="${bestOffer.url}" target="_blank">${bestOffer.url}</a></div>`;
  }
  
  if (bestOffer.price_str) {
    html += `<div style="font-size: 14px; margin: 8px 0;">💰 Цена: <strong>${bestOffer.price_str}</strong></div>`;
  }
  
  if (bestOffer.year) {
    html += `<div style="font-size: 13px; color: var(--muted);">📅 Год: ${bestOffer.year}</div>`;
  }
  
  if (bestOffer.condition) {
    html += `<div style="font-size: 13px; color: var(--muted);">⚙️ Состояние: ${bestOffer.condition}</div>`;
  }
  
  const score = analysis.best_score || 0;
  html += `<div class="best-offer-score">⭐ Оценка: ${score.toFixed(1)}/10</div>`;
  
  if (analysis.reason) {
    html += `<div class="best-offer-reason">💡 ${analysis.reason}</div>`;
  }
  
  card.innerHTML = html;
}

// ===== RENDER BEST OFFERS COMPARISON (КАРУСЕЛЬ) =====
function renderBestComparison(data) {
  const section = document.getElementById("bestComparisonSection");
  const content = document.getElementById("bestComparisonContent");
  
  if (!section || !content) return;
  
  const comparisons = data.best_offers_comparison || {};
  
  if (Object.keys(comparisons).length === 0) {
    section.classList.add("hidden");
    return;
  }
  
  section.classList.remove("hidden");
  content.innerHTML = "";
  
  // Сохраняем данные для карусели
  window.lastAnalysisData = data;
  window.comparisonsArray = Object.entries(comparisons);
  window.currentComparisonIndex = 0;
  
  // Показываем первый элемент
  renderComparisonSlide();
  
  // Обновляем счётчик
  updateComparisonCounter();
}

// Отображение одного слайда карусели
function renderComparisonSlide() {
  const content = document.getElementById("bestComparisonContent");
  if (!content || !window.comparisonsArray || window.comparisonsArray.length === 0) return;
  
  const [analogName, comp] = window.comparisonsArray[window.currentComparisonIndex];
  const data = window.lastAnalysisData;
  
  content.innerHTML = `<h3 style="margin: 0 0 12px 0; font-size: 16px; color: var(--text); font-weight: 600; padding-bottom: 12px; border-bottom: 1px solid var(--border);">🎪 Сравнение лучших объявлений</h3>`;
  
  const div = document.createElement("div");
  div.className = "comparison-item";
  
  let html = `<div class="comparison-header">`;
  html += `<div><strong>${data.vendor || "Оригинал"}</strong> vs <strong>${analogName}</strong></div>`;
  html += `<div class="comparison-winner">🏆 ${comp.winner === "original" ? "Оригинал" : "Аналог"}</div>`;
  html += `</div>`;
  
  html += `<div class="comparison-scores">`;
  html += `<div class="comparison-score">Оригинал: ${(comp.original_score || 0).toFixed(1)}/10</div>`;
  html += `<div class="comparison-score">Аналог: ${(comp.analog_score || 0).toFixed(1)}/10</div>`;
  html += `</div>`;
  
  // Links to offers
  html += `<div class="comparison-links">`;
  if (comp.original_url) {
    html += `<div class="offer-link">`;
    html += `<strong>🔗 Оригинал:</strong> `;
    html += `<a href="${comp.original_url}" target="_blank" rel="noopener noreferrer">${comp.original_title || comp.original_url}</a>`;
    html += `</div>`;
  }
  if (comp.analog_url) {
    html += `<div class="offer-link">`;
    html += `<strong>🔗 Аналог:</strong> `;
    html += `<a href="${comp.analog_url}" target="_blank" rel="noopener noreferrer">${comp.analog_title || comp.analog_url}</a>`;
    html += `</div>`;
  }
  html += `</div>`;
  
  // Detailed comparison
  if (comp.comparison_details) {
    html += `<div class="comparison-details">`;
    html += `<h4>📊 Детальное сравнение:</h4>`;
    if (comp.comparison_details.price) {
      html += `<div class="detail-item"><strong>💰 Цена:</strong> ${comp.comparison_details.price}</div>`;
    }
    if (comp.comparison_details.quality) {
      html += `<div class="detail-item"><strong>⚙️ Качество:</strong> ${comp.comparison_details.quality}</div>`;
    }
    if (comp.comparison_details.financing) {
      html += `<div class="detail-item"><strong>💳 Финансирование:</strong> ${comp.comparison_details.financing}</div>`;
    }
    if (comp.comparison_details.reliability) {
      html += `<div class="detail-item"><strong>🛡️ Надежность:</strong> ${comp.comparison_details.reliability}</div>`;
    }
    if (comp.comparison_details.value) {
      html += `<div class="detail-item"><strong>⭐ Ценность:</strong> ${comp.comparison_details.value}</div>`;
    }
    html += `</div>`;
  }
  
  // Key differences
  if (comp.key_differences && comp.key_differences.length > 0) {
    html += `<div class="key-differences">`;
    html += `<h4>🔑 Ключевые отличия:</h4>`;
    html += `<ul>`;
    comp.key_differences.forEach(diff => {
      html += `<li>${diff}</li>`;
    });
    html += `</ul></div>`;
  }
  
  // Price comparison
  if (comp.price_comparison) {
    const pc = comp.price_comparison;
    const origPrice = pc.original_price ? pc.original_price.toLocaleString("ru-RU") + " ₽" : "—";
    const analogPrice = pc.analog_price ? pc.analog_price.toLocaleString("ru-RU") + " ₽" : "—";
    const diff = pc.difference_percent ? `${pc.difference_percent > 0 ? "+" : ""}${pc.difference_percent.toFixed(1)}%` : "";
    
    html += `<div class="comparison-price">`;
    html += `<strong>💰 Сравнение цен:</strong><br>`;
    html += `Оригинал: ${origPrice}`;
    if (pc.monthly_payment_original) {
      html += ` (${pc.monthly_payment_original.toLocaleString("ru-RU")} ₽/мес)`;
    }
    html += `<br>Аналог: ${analogPrice}`;
    if (pc.monthly_payment_analog) {
      html += ` (${pc.monthly_payment_analog.toLocaleString("ru-RU")} ₽/мес)`;
    }
    if (diff) {
      html += `<br>Разница: <strong>${diff}</strong>`;
    }
    html += `</div>`;
  }
  
  // Pros and cons
  html += `<div class="comparison-pros-cons">`;
  
  if (comp.pros_original && comp.pros_original.length > 0) {
    html += `<div class="comparison-pros">`;
    html += `<h4 style="color: var(--accent);">✅ Плюсы оригинала</h4>`;
    html += `<ul>`;
    comp.pros_original.slice(0, 3).forEach(p => {
      html += `<li>+ ${p}</li>`;
    });
    html += `</ul></div>`;
  }
  
  if (comp.cons_original && comp.cons_original.length > 0) {
    html += `<div class="comparison-cons">`;
    html += `<h4 style="color: var(--danger);">❌ Минусы оригинала</h4>`;
    html += `<ul>`;
    comp.cons_original.slice(0, 3).forEach(c => {
      html += `<li>- ${c}</li>`;
    });
    html += `</ul></div>`;
  }
  
  if (comp.pros_analog && comp.pros_analog.length > 0) {
    html += `<div class="comparison-pros">`;
    html += `<h4 style="color: var(--accent);">✅ Плюсы аналога</h4>`;
    html += `<ul>`;
    comp.pros_analog.slice(0, 3).forEach(p => {
      html += `<li>+ ${p}</li>`;
    });
    html += `</ul></div>`;
  }
  
  if (comp.cons_analog && comp.cons_analog.length > 0) {
    html += `<div class="comparison-cons">`;
    html += `<h4 style="color: var(--danger);">❌ Минусы аналога</h4>`;
    html += `<ul>`;
    comp.cons_analog.slice(0, 3).forEach(c => {
      html += `<li>- ${c}</li>`;
    });
    html += `</ul></div>`;
  }
  
  html += `</div>`;
  
  // Recommendation
  if (comp.recommendation) {
    html += `<div class="comparison-recommendation">`;
    html += `<strong>💡 Рекомендация:</strong><br>${comp.recommendation}`;
    html += `</div>`;
  }
  
  div.innerHTML = html;
  content.appendChild(div);
  
  // Обновляем состояние кнопок
  const prevBtn = document.getElementById("prevComparisonBtn");
  const nextBtn = document.getElementById("nextComparisonBtn");
  if (prevBtn) prevBtn.disabled = window.currentComparisonIndex === 0;
  if (nextBtn) nextBtn.disabled = window.currentComparisonIndex === window.comparisonsArray.length - 1;
}

// Обновляем счётчик
function updateComparisonCounter() {
  const counter = document.getElementById("comparisonCounter");
  if (counter) {
    const total = window.comparisonsArray ? window.comparisonsArray.length : 0;
    const current = total > 0 ? window.currentComparisonIndex + 1 : 0;
    counter.textContent = `${current}/${total}`;
  }
}

// Навигация по карусели
function nextComparisonSlide() {
  if (window.currentComparisonIndex < window.comparisonsArray.length - 1) {
    window.currentComparisonIndex++;
    renderComparisonSlide();
    updateComparisonCounter();
  }
}

function prevComparisonSlide() {
  if (window.currentComparisonIndex > 0) {
    window.currentComparisonIndex--;
    renderComparisonSlide();
    updateComparisonCounter();
  }
}

// Инициализация кнопок карусели сравнений
document.addEventListener("DOMContentLoaded", function() {
  const prevBtn = document.getElementById("prevComparisonBtn");
  const nextBtn = document.getElementById("nextComparisonBtn");
  
  if (prevBtn) prevBtn.addEventListener("click", prevComparisonSlide);
  if (nextBtn) nextBtn.addEventListener("click", nextComparisonSlide);
});

// ===== КАРУСЕЛЬ СРАВНЕНИЙ =====
let currentComparisonIndex = 0;
let comparisonsArray = [];

function initComparisonCarousel(data) {
  comparisonsArray = Object.entries(data.best_offers_comparison || {});
  currentComparisonIndex = 0;
  
  if (comparisonsArray.length > 0) {
    renderComparisonCarousel();
  }
}

function renderComparisonCarousel() {
  const content = document.getElementById("bestComparisonContent");
  if (!content || comparisonsArray.length === 0) return;

  const [analogName, comp] = comparisonsArray[currentComparisonIndex];
  const data = window.lastAnalysisData; // сохраняем данные глобально

  content.innerHTML = `<h3 style="margin: 0 0 12px 0; font-size: 16px; color: var(--text); font-weight: 600; padding-bottom: 12px; border-bottom: 1px solid var(--border);">🎪 Сравнение лучших объявлений</h3>`;

  const div = document.createElement("div");
  div.className = "comparison-item";

  let html = `<div class="comparison-header">
    <div><strong>${data.vendor || "Оригинал"}</strong> vs <strong>${analogName}</strong></div>
    <div class="comparison-winner">${comp.winner === "original" ? "✅ Оригинал" : "🏆 " + analogName}</div>
  </div>`;

  html += `<div class="comparison-scores">
    <div class="comparison-score">Оригинал: ${comp.original_score ? comp.original_score.toFixed(1) : 0}</div>
    <div class="comparison-score">Аналог: ${comp.analog_score ? comp.analog_score.toFixed(1) : 0}</div>
  </div>`;

  // Links
  html += `<div class="comparison-links">`;
  if (comp.original_url) {
    html += `<div class="offer-link"><strong>🔗 Оригинал:</strong> <a href="${comp.original_url}" target="_blank">${comp.original_title || comp.original_url}</a></div>`;
  }
  if (comp.analog_url) {
    html += `<div class="offer-link"><strong>🔗 Аналог:</strong> <a href="${comp.analog_url}" target="_blank">${comp.analog_title || comp.analog_url}</a></div>`;
  }
  html += `</div>`;

  // Детали, различия, цена, плюсы/минусы
  if (comp.comparison_details) {
    html += `<div class="comparison-details"><h4>📊 Сравнение</h4>`;
    if (comp.comparison_details.price) html += `<div class="detail-item"><strong>Цена:</strong> ${comp.comparison_details.price}</div>`;
    if (comp.comparison_details.quality) html += `<div class="detail-item"><strong>Качество:</strong> ${comp.comparison_details.quality}</div>`;
    html += `</div>`;
  }

  if (comp.key_differences && comp.key_differences.length > 0) {
    html += `<div class="key-differences"><h4>⭐ Отличия</h4><ul>`;
    comp.key_differences.forEach(d => html += `<li>${d}</li>`);
    html += `</ul></div>`;
  }

  if (comp.price_comparison) {
    const pc = comp.price_comparison;
    html += `<div class="comparison-price"><strong>💰 Цена:</strong> ${pc.original_price ? pc.original_price.toLocaleString("ru-RU") : "—"} vs ${pc.analog_price ? pc.analog_price.toLocaleString("ru-RU") : "—"}</div>`;
  }

  html += `<div class="comparison-pros-cons">`;
  if (comp.pros_original?.length) {
    html += `<div class="comparison-pros"><h4>✅ Плюсы оригинала</h4><ul>`;
    comp.pros_original.slice(0, 3).forEach(p => html += `<li>✓ ${p}</li>`);
    html += `</ul></div>`;
  }
  if (comp.cons_original?.length) {
    html += `<div class="comparison-cons"><h4>❌ Минусы оригинала</h4><ul>`;
    comp.cons_original.slice(0, 3).forEach(c => html += `<li>✗ ${c}</li>`);
    html += `</ul></div>`;
  }
  html += `</div>`;

  if (comp.recommendation) {
    html += `<div class="comparison-recommendation"><strong>💡 Рекомендация:</strong><br>${comp.recommendation}</div>`;
  }

  div.innerHTML = html;
  content.appendChild(div);

  // Счётчик
  const counter = document.getElementById("comparisonCounter");
  if (counter) {
    counter.textContent = `${currentComparisonIndex + 1}/${comparisonsArray.length}`;
  }
}

function nextComparison() {
  if (currentComparisonIndex < comparisonsArray.length - 1) {
    currentComparisonIndex++;
    renderComparisonCarousel();
  }
}

function prevComparison() {
  if (currentComparisonIndex > 0) {
    currentComparisonIndex--;
    renderComparisonCarousel();
  }
}

// Инициализация кнопок карусели
document.addEventListener("DOMContentLoaded", function() {
  const prevCompBtn = document.getElementById("prevComparisonBtn");
  const nextCompBtn = document.getElementById("nextComparisonBtn");
  
  if (prevCompBtn) prevCompBtn.addEventListener("click", prevComparison);
  if (nextCompBtn) nextCompBtn.addEventListener("click", nextComparison);
});

// ===== RENDER ALL OFFERS =====
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
    
    let html = `<div class="offer-item-header">`;
    html += `<span class="offer-number">#${index + 1}</span>`;
    if (offer.url) {
      html += `<a href="${offer.url}" target="_blank" rel="noopener noreferrer" class="offer-title-link">${offer.title || "Объявление"}</a>`;
    } else {
      html += `<span class="offer-title">${offer.title || "Объявление"}</span>`;
    }
    html += `</div>`;
    
    html += `<div class="offer-item-details">`;
    if (offer.source) {
      html += `<span class="offer-source">📍 ${offer.source}</span>`;
    }
    if (offer.price_str) {
      html += `<span class="offer-price">💰 ${offer.price_str}</span>`;
    }
    if (offer.monthly_payment_str) {
      html += `<span class="offer-payment">💳 ${offer.monthly_payment_str}/мес</span>`;
    }
    if (offer.year) {
      html += `<span class="offer-year">📅 ${offer.year}</span>`;
    }
    if (offer.condition) {
      html += `<span class="offer-condition">⚙️ ${offer.condition}</span>`;
    }
    if (offer.location) {
      html += `<span class="offer-location">🌍 ${offer.location}</span>`;
    }
    html += `</div>`;
    
    div.innerHTML = html;
    list.appendChild(div);
  });
  
  // Update title
  const titleEl = document.getElementById("allOffersTitle");
  if (titleEl) {
    titleEl.textContent = `Все объявления (${sources.length})`;
  }
  
  // Toggle button handler
  toggleBtn.onclick = () => {
    const isHidden = section.classList.contains("hidden");
    if (isHidden) {
      section.classList.remove("hidden");
      toggleBtn.textContent = "📋 Скрыть все объявления";
    } else {
      section.classList.add("hidden");
      toggleBtn.textContent = `📋 Показать все объявления (${sources.length})`;
    }
  };
  
  // Update button text
  toggleBtn.textContent = `📋 Показать все объявления (${sources.length})`;
}
