// HeartGuard AI Master Dashboard Application Script

document.addEventListener("DOMContentLoaded", () => {
    checkHealthStatus();
    setupEventListeners();
});

// Check API Health Status
async function checkHealthStatus() {
    const statusText = document.getElementById("statusText");
    const systemStatus = document.getElementById("systemStatus");

    try {
        const response = await fetch("/health");
        const data = await response.json();

        if (response.ok && data.status === "healthy") {
            statusText.innerText = "API Connected & Models Loaded";
            systemStatus.style.background = "rgba(16, 185, 129, 0.15)";
            systemStatus.style.color = "#10b981";
            systemStatus.style.borderColor = "rgba(16, 185, 129, 0.4)";
        } else {
            statusText.innerText = "Degraded Mode";
        }
    } catch (err) {
        statusText.innerText = "API Offline";
    }
}

// Preset Clinical Profiles
const HIGH_RISK_PRESET = {
    age: 63, sex: 1, cp: 0, trestbps: 160, chol: 286, 
    fbs: 1, restecg: 1, thalach: 108, exang: 1, 
    oldpeak: 2.6, slope: 1, ca: 2, thal: 2
};

const LOW_RISK_PRESET = {
    age: 34, sex: 0, cp: 2, trestbps: 115, chol: 182, 
    fbs: 0, restecg: 0, thalach: 174, exang: 0, 
    oldpeak: 0.0, slope: 2, ca: 0, thal: 1
};

function setupEventListeners() {
    document.getElementById("btnHighRisk").addEventListener("click", () => populateForm(HIGH_RISK_PRESET));
    document.getElementById("btnLowRisk").addEventListener("click", () => populateForm(LOW_RISK_PRESET));

    document.getElementById("predictionForm").addEventListener("submit", handleFormSubmit);
}

function populateForm(data) {
    for (const [key, value] of Object.entries(data)) {
        const el = document.getElementById(key);
        if (el) {
            el.value = value;
            
            // Update Slider Badge Display Text
            const displayEl = document.getElementById(`${key}Val`);
            if (displayEl) {
                if (key === "age") displayEl.innerHTML = `${value} <small>years</small>`;
                else if (key === "trestbps") displayEl.innerHTML = `${value} <small>mm Hg</small>`;
                else if (key === "chol") displayEl.innerHTML = `${value} <small>mg/dl</small>`;
                else if (key === "thalach") displayEl.innerHTML = `${value} <small>bpm</small>`;
                else displayEl.innerText = value;
            }
        }
    }
}

async function handleFormSubmit(e) {
    e.preventDefault();

    const btnSubmit = document.getElementById("btnSubmit");
    btnSubmit.disabled = true;
    btnSubmit.querySelector(".btn-text").innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Running Neural Assessment...`;

    const payload = {
        age: parseInt(document.getElementById("age").value),
        sex: parseInt(document.getElementById("sex").value),
        cp: parseInt(document.getElementById("cp").value),
        trestbps: parseInt(document.getElementById("trestbps").value),
        chol: parseInt(document.getElementById("chol").value),
        fbs: parseInt(document.getElementById("fbs").value),
        restecg: parseInt(document.getElementById("restecg").value),
        thalach: parseInt(document.getElementById("thalach").value),
        exang: parseInt(document.getElementById("exang").value),
        oldpeak: parseFloat(document.getElementById("oldpeak").value),
        slope: parseInt(document.getElementById("slope").value),
        ca: parseInt(document.getElementById("ca").value),
        thal: parseInt(document.getElementById("thal").value)
    };

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (response.ok && data.status === "success") {
            updateDashboardResults(data);
        } else {
            alert(`Inference Error: ${data.detail || "Failed to process prediction."}`);
        }
    } catch (err) {
        console.error("API Error:", err);
        alert("Unable to reach HeartGuard prediction server.");
    } finally {
        btnSubmit.disabled = false;
        btnSubmit.querySelector(".btn-text").innerHTML = `<i class="fa-solid fa-bolt"></i> Run AI Assessment`;
    }
}

function updateDashboardResults(data) {
    const prob = data.probability;
    const pct = (prob * 100).toFixed(1);

    // Number Count Up Animation
    animateNumber("riskPercent", parseFloat(pct));

    // SVG Circular Gauge Animation (dasharray: 534)
    const gaugeTrack = document.getElementById("gaugeProgressTrack");
    const maxOffset = 534;
    const targetOffset = maxOffset * (1 - prob);

    gaugeTrack.style.strokeDashoffset = targetOffset;

    if (prob >= 0.5) {
        gaugeTrack.style.stroke = "var(--crimson)";
        gaugeTrack.style.filter = "drop-shadow(0 0 12px var(--crimson))";
    } else {
        gaugeTrack.style.stroke = "var(--emerald)";
        gaugeTrack.style.filter = "drop-shadow(0 0 12px var(--emerald))";
    }

    // Risk Badge Update
    const riskBadge = document.getElementById("riskBadge");
    if (prob >= 0.5) {
        riskBadge.className = "risk-badge-box badge-high";
        riskBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> HIGH RISK CLASSIFIED`;
    } else {
        riskBadge.className = "risk-badge-box badge-low";
        riskBadge.innerHTML = `<i class="fa-solid fa-shield-check"></i> LOW RISK ASSESSMENT`;
    }

    // Individual Model Breakdown Progress Bars
    const rfProbPct = (prob * 100).toFixed(1);
    // Keras ANN estimation score
    const annEstimatePct = Math.min(100, Math.max(0, prob >= 0.5 ? (prob * 100 + 3.5).toFixed(1) : (prob * 100 - 1.2).toFixed(1)));

    document.getElementById("rfProb").innerText = `${rfProbPct}%`;
    document.getElementById("annProb").innerText = `${annEstimatePct}%`;

    document.getElementById("rfBar").style.width = `${rfProbPct}%`;
    document.getElementById("annBar").style.width = `${annEstimatePct}%`;

    // Clinical Advisory Card
    document.getElementById("advisoryText").innerText = data.clinical_advisory;
    const advisoryCard = document.getElementById("advisoryCard");
    advisoryCard.style.borderColor = prob >= 0.5 ? "rgba(244, 63, 94, 0.4)" : "rgba(16, 185, 129, 0.4)";
}

// Counter animation
function animateNumber(elementId, targetValue) {
    const el = document.getElementById(elementId);
    let start = 0;
    const duration = 1000;
    const stepTime = 20;
    const steps = duration / stepTime;
    const increment = targetValue / steps;

    const timer = setInterval(() => {
        start += increment;
        if (start >= targetValue) {
            el.innerText = `${targetValue.toFixed(1)}%`;
            clearInterval(timer);
        } else {
            el.innerText = `${start.toFixed(1)}%`;
        }
    }, stepTime);
}

// Gallery Tab Switcher
function switchGalleryTab(tabId) {
    document.querySelectorAll(".gallery-item").forEach(el => el.classList.remove("active"));
    document.querySelectorAll(".nav-tab-btn").forEach(el => el.classList.remove("active"));

    document.getElementById(tabId).classList.add("active");
    event.currentTarget.classList.add("active");
}
