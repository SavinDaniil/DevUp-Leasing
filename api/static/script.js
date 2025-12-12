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
    renderSources(data.sources || []);   // новая строка

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
