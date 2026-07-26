// Heart Guard MLOps Web Dashboard Application Script

document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
    setupEventListeners();
});

// Check API Health Endpoint
async function checkHealth() {
    const statusText = document.getElementById("statusText");
    const systemStatus = document.getElementById("systemStatus");

    try {
        const response = await fetch("/health");
        const data = await response.json();

        if (response.ok && data.status === "healthy") {
            statusText.innerText = "System Online & Models Loaded";
            systemStatus.style.background = "rgba(16, 185, 129, 0.15)";
            systemStatus.style.color = "#10b981";
        } else {
            statusText.innerText = "System Degraded";
        }
    } catch (err) {
        statusText.innerText = "API Offline (Local Model Mode)";
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
    // Preset Buttons
    document.getElementById("btnHighRisk").addEventListener("click", () => populateForm(HIGH_RISK_PRESET));
    document.getElementById("btnLowRisk").addEventListener("click", () => populateForm(LOW_RISK_PRESET));

    // Form Submit Listener
    document.getElementById("predictionForm").addEventListener("submit", handleFormSubmit);
}

// Populate form fields with preset values
function populateForm(data) {
    for (const [key, value] of Object.entries(data)) {
        const el = document.getElementById(key);
        if (el) {
            el.value = value;
            // Trigger slider display text update
            const displayEl = document.getElementById(`${key}Val`);
            if (displayEl) {
                displayEl.innerText = value;
            }
        }
    }
}

// Handle Form Submission & Predict API Call
async function handleFormSubmit(e) {
    e.preventDefault();

    const btnSubmit = document.getElementById("btnSubmit");
    btnSubmit.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing Inference...`;
    btnSubmit.disabled = true;

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
            updateResultCard(data);
        } else {
            alert(`Inference Error: ${data.detail || "Failed to process prediction."}`);
        }
    } catch (err) {
        console.error("API Call Error:", err);
        alert("Error connecting to FastAPI prediction service.");
    } finally {
        btnSubmit.innerHTML = `<i class="fa-solid fa-stethoscope"></i> Calculate Risk Assessment`;
        btnSubmit.disabled = false;
    }
}

// Update Result Card & Animate Gauge Meter
function updateResultCard(data) {
    const prob = data.probability;
    const pct = (prob * 100).toFixed(1);

    // Update Percentage
    document.getElementById("riskPercent").innerText = `${pct}%`;
    document.getElementById("probScore").innerText = `${data.probability} (${data.risk_score_pct})`;

    // Gauge Conic Gradient Animation
    const gaugeFill = document.getElementById("gaugeFill");
    const angle = (prob * 360).toFixed(0);
    const color = prob >= 0.5 ? "#ef4444" : "#10b981";
    gaugeFill.style.background = `conic-gradient(${color} ${angle}deg, rgba(255, 255, 255, 0.05) ${angle}deg)`;

    // Update Risk Badge
    const riskBadge = document.getElementById("riskBadge");
    if (prob >= 0.5) {
        riskBadge.className = "risk-badge badge-high";
        riskBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> HIGH RISK DETECTED`;
    } else {
        riskBadge.className = "risk-badge badge-low";
        riskBadge.innerHTML = `<i class="fa-solid fa-shield-heart"></i> LOW RISK ASSESSMENT`;
    }

    // Update Advisory Box
    const advisoryText = document.getElementById("advisoryText");
    advisoryText.innerText = data.clinical_advisory;
}

// Analytics Tab Switcher
function showTab(tabId) {
    document.querySelectorAll(".tab-content").forEach(el => el.classList.remove("active"));
    document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));

    document.getElementById(tabId).classList.add("active");
    event.target.classList.add("active");
}
