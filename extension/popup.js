const API_URL = "https://fact-check-ml-api.onrender.com/predict";

const checkBtn = document.getElementById("check-btn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const labelEl = document.getElementById("label");
const confidenceEl = document.getElementById("confidence");
const termsEl = document.getElementById("terms");
const explanationNoteEl = document.getElementById("explanation-note");
const errorEl = document.getElementById("error");

// Injected into the active tab via chrome.scripting.executeScript, so it
// must be fully self-contained (no references to popup.js's own scope).
function extractPageContent() {
  const title = document.title || "";
  let text = "";
  const article = document.querySelector("article");
  if (article && article.innerText.trim().length > 200) {
    text = article.innerText;
  } else {
    const paragraphs = Array.from(document.querySelectorAll("p"))
      .map((p) => p.innerText.trim())
      .filter((t) => t.length > 40);
    text = paragraphs.join(" ");
  }
  // Cap payload size -- the model only needs a representative sample.
  return { title, text: text.slice(0, 5000) };
}

function setLoading(isLoading) {
  checkBtn.disabled = isLoading;
  statusEl.hidden = !isLoading;
  resultEl.hidden = true;
  errorEl.hidden = true;
}

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = false;
  resultEl.hidden = true;
}

function renderResult(data) {
  labelEl.textContent = data.label;
  labelEl.className = "badge " + (data.label === "REAL" ? "badge-real" : "badge-fake");
  confidenceEl.textContent = `${data.confidence.toFixed(1)}% confidence`;

  termsEl.innerHTML = "";
  for (const t of data.top_terms) {
    const li = document.createElement("li");
    li.textContent = t.term;
    // Only "coefficient" explanations have a meaningful per-word sign
    // (TF-IDF weights used for "salience" are always >= 0, so coloring
    // by sign there would just paint every term green).
    li.className =
      data.explanation_type === "coefficient"
        ? t.weight >= 0
          ? "term-real"
          : "term-fake"
        : "term-neutral";
    termsEl.appendChild(li);
  }

  explanationNoteEl.textContent =
    data.explanation_type === "coefficient"
      ? "Terms above are the words that most pushed this prediction toward REAL (green) or FAKE (red)."
      : "This model can't attribute direction per word -- terms above are just the most notable words in the text.";

  resultEl.hidden = false;
}

checkBtn.addEventListener("click", async () => {
  setLoading(true);
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !tab.id) {
      throw new Error("No active tab found.");
    }

    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractPageContent,
    });

    if (!result.title && !result.text) {
      throw new Error("Couldn't find any article text on this page.");
    }

    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(result),
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${response.status})`);
    }

    const data = await response.json();
    renderResult(data);
  } catch (err) {
    showError(
      err.message.includes("Failed to fetch")
        ? "Couldn't reach the API server. If it's been idle a while (free tier spins down), it may just be waking up -- try again in ~30 seconds."
        : err.message
    );
  } finally {
    statusEl.hidden = true;
    checkBtn.disabled = false;
  }
});
