/**
 * dashboard/static/app.js
 * Frontend logic for PotholeOps Dashboard
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const tabs = document.querySelectorAll(".nav-tab");
  const tabContents = document.querySelectorAll(".tab-content");
  
  const apiStatusPill = document.getElementById("apiStatusPill");
  const apiStatusText = document.getElementById("apiStatusText");
  const refreshHealthBtn = document.getElementById("refreshHealthBtn");
  
  // Drag & Drop / Upload elements
  const dropZone = document.getElementById("dropZone");
  const imageInput = document.getElementById("imageInput");
  const dropZonePrompt = document.getElementById("dropZonePrompt");
  const previewWrapper = document.getElementById("previewWrapper");
  const imagePreview = document.getElementById("imagePreview");
  const removeImgBtn = document.getElementById("removeImgBtn");
  const fileInfo = document.getElementById("fileInfo");
  const fileName = document.getElementById("fileName");
  const fileSize = document.getElementById("fileSize");
  const predictBtn = document.getElementById("predictBtn");
  const btnSpinner = document.getElementById("btnSpinner");
  const btnText = document.getElementById("btnText");
  
  // Inference Results elements
  const severityBadge = document.getElementById("severityBadge");
  const confidenceValue = document.getElementById("confidenceValue");
  const confidenceBar = document.getElementById("confidenceBar");
  
  // Drift elements
  const bannerTitle = document.getElementById("bannerTitle");
  const bannerDesc = document.getElementById("bannerDesc");
  const triggerDriftBtn = document.getElementById("triggerDriftBtn");
  const driftSpinner = document.getElementById("driftSpinner");
  const refreshDriftBtn = document.getElementById("refreshDriftBtn");
  const driftIframe = document.getElementById("driftIframe");
  
  // Logs elements
  const refreshLogsBtn = document.getElementById("refreshLogsBtn");
  const logsTableBody = document.getElementById("logsTableBody");

  // Review Queue elements
  const refreshReviewBtn = document.getElementById("refreshReviewBtn");
  const reviewGrid = document.getElementById("reviewGrid");

  let selectedFile = null;
  let probsChartInstance = null;
  let refChartInstance = null;
  let curChartInstance = null;

  // --- 1. Tab Navigation ---
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));
      
      tab.classList.add("active");
      const target = document.getElementById(tab.dataset.tab);
      if (target) target.classList.add("active");

      if (tab.dataset.tab === "tab-drift") {
        fetchDriftSummary();
      } else if (tab.dataset.tab === "tab-logs") {
        fetchLogs();
      } else if (tab.dataset.tab === "tab-review") {
        fetchReviewQueue();
      }
    });
  });

  // --- 2. API Health Check ---
  async function checkHealth() {
    apiStatusText.textContent = "Checking API...";
    apiStatusPill.className = "status-pill";
    try {
      const res = await fetch("/health");
      const data = await res.json();
      if (data.status === "ok") {
        apiStatusText.textContent = "API ONLINE (port 8000)";
        apiStatusPill.className = "status-pill online";
      } else {
        throw new Error("Invalid status");
      }
    } catch (err) {
      apiStatusText.textContent = "API OFFLINE";
      apiStatusPill.className = "status-pill offline";
    }
  }

  refreshHealthBtn.addEventListener("click", checkHealth);
  checkHealth();

  // --- 3. Drag and Drop File Handlers ---
  dropZone.addEventListener("click", (e) => {
    if (e.target !== removeImgBtn && !selectedFile) {
      imageInput.click();
    }
  });

  ["dragenter", "dragover"].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  imageInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please select a valid image file (.jpg, .jpeg, .png).");
      return;
    }

    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;

    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      dropZonePrompt.classList.add("hidden");
      previewWrapper.classList.remove("hidden");
      fileInfo.classList.remove("hidden");
      predictBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  removeImgBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    selectedFile = null;
    imageInput.value = "";
    previewWrapper.classList.add("hidden");
    dropZonePrompt.classList.remove("hidden");
    fileInfo.classList.add("hidden");
    predictBtn.disabled = true;
    resetResults();
  });

  function resetResults() {
    severityBadge.className = "severity-badge severity-placeholder";
    severityBadge.textContent = "Awaiting Image";
    confidenceValue.textContent = "0.0%";
    confidenceBar.style.width = "0%";
    if (probsChartInstance) {
      probsChartInstance.data.datasets[0].data = [0, 0, 0];
      probsChartInstance.update();
    }
  }

  // --- 4. Chart.js Initialization ---
  function initProbsChart() {
    const ctx = document.getElementById("probsChart").getContext("2d");
    probsChartInstance = new Chart(ctx, {
      type: "bar",
      data: {
        labels: ["Low", "Medium", "High"],
        datasets: [{
          label: "Probability",
          data: [0, 0, 0],
          backgroundColor: ["#10b981", "#f59e0b", "#ef4444"],
          borderRadius: 6,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 0,
            max: 1,
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#94a3b8" }
          },
          x: {
            grid: { display: false },
            ticks: { color: "#94a3b8", font: { weight: "500" } }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `Probability: ${(ctx.raw * 100).toFixed(1)}%`
            }
          }
        }
      }
    });
  }

  initProbsChart();

  // --- 5. Run Prediction ---
  predictBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    predictBtn.disabled = true;
    btnSpinner.classList.remove("hidden");
    btnText.textContent = "Classifying...";

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch("/predict", {
        method: "POST",
        body: formData
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();
      displayPredictionResult(data);

    } catch (err) {
      alert(`Prediction failed: ${err.message}`);
    } finally {
      predictBtn.disabled = false;
      btnSpinner.classList.add("hidden");
      btnText.textContent = "Classify Pothole Severity";
    }
  });

  function displayPredictionResult(data) {
    const pred = data.prediction.toLowerCase();
    const conf = (data.confidence * 100).toFixed(1);

    severityBadge.className = `severity-badge ${pred}`;
    severityBadge.textContent = pred.toUpperCase();

    confidenceValue.textContent = `${conf}%`;
    confidenceBar.style.width = `${conf}%`;

    const probs = data.class_probabilities;
    const probValues = [probs.low || 0, probs.medium || 0, probs.high || 0];

    probsChartInstance.data.datasets[0].data = probValues;
    probsChartInstance.update();
  }

  // --- 6. Drift Monitoring ---
  async function fetchDriftSummary() {
    try {
      const res = await fetch("/reports/drift_summary.json");
      if (!res.ok) throw new Error("No drift summary JSON found.");

      const data = await res.json();
      renderDriftSummary(data);
    } catch (err) {
      bannerTitle.textContent = "No Drift Summary Yet";
      bannerDesc.textContent = "Click 'Run Fresh Drift Check' below to generate Evidently AI drift evaluation.";
    }
  }

  function renderDriftSummary(data) {
    const isDrift = data.drift_detected;
    if (isDrift) {
      bannerTitle.textContent = "DATA DRIFT DETECTED";
      bannerDesc.textContent = "Current prediction distribution deviates significantly from validation baseline.";
      bannerTitle.style.color = "#ef4444";
    } else {
      bannerTitle.textContent = "NO SIGNIFICANT DRIFT DETECTED";
      bannerDesc.textContent = "Current live predictions align well with validation baseline distribution.";
      bannerTitle.style.color = "#10b981";
    }

    renderDriftCharts(data.reference_counts || {}, data.current_counts || {});
  }

  function renderDriftCharts(refCounts, curCounts) {
    const classes = ["low", "medium", "high"];
    const refData = classes.map(c => refCounts[c] || 0);
    const curData = classes.map(c => curCounts[c] || 0);

    // Reference chart
    const ctxRef = document.getElementById("refChart").getContext("2d");
    if (refChartInstance) refChartInstance.destroy();
    refChartInstance = new Chart(ctxRef, {
      type: "bar",
      data: {
        labels: ["Low", "Medium", "High"],
        datasets: [{
          label: "Baseline Count",
          data: refData,
          backgroundColor: ["#10b981", "#f59e0b", "#ef4444"],
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } }, x: { ticks: { color: "#94a3b8" } } }
      }
    });

    // Current chart
    const ctxCur = document.getElementById("curChart").getContext("2d");
    if (curChartInstance) curChartInstance.destroy();
    curChartInstance = new Chart(ctxCur, {
      type: "bar",
      data: {
        labels: ["Low", "Medium", "High"],
        datasets: [{
          label: "Current Count",
          data: curData,
          backgroundColor: ["#10b981", "#f59e0b", "#ef4444"],
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } }, x: { ticks: { color: "#94a3b8" } } }
      }
    });
  }

  triggerDriftBtn.addEventListener("click", async () => {
    triggerDriftBtn.disabled = true;
    driftSpinner.classList.remove("hidden");

    try {
      await fetch("/api/trigger-drift", { method: "POST" });
      bannerTitle.textContent = "Running Drift Evaluation...";
      bannerDesc.textContent = "Executing Evidently AI background evaluation...";
      
      // Poll after 4 seconds
      setTimeout(() => {
        fetchDriftSummary();
        driftIframe.src = driftIframe.src; // Reload iframe
      }, 4000);
    } catch (err) {
      alert(`Drift evaluation failed: ${err.message}`);
    } finally {
      triggerDriftBtn.disabled = false;
      driftSpinner.classList.add("hidden");
    }
  });

  refreshDriftBtn.addEventListener("click", () => {
    fetchDriftSummary();
    driftIframe.src = driftIframe.src;
  });

  // --- 7. Prediction History Logs ---
  async function fetchLogs() {
    try {
      const res = await fetch("/api/logs");
      const logs = await res.json();
      renderLogsTable(logs);
    } catch (err) {
      logsTableBody.innerHTML = `<tr><td colspan="5" class="table-empty">Error loading logs.</td></tr>`;
    }
  }

  refreshLogsBtn.addEventListener("click", fetchLogs);

  function renderLogsTable(logs) {
    if (!logs || logs.length === 0) {
      logsTableBody.innerHTML = `<tr><td colspan="5" class="table-empty">No prediction records logged yet.</td></tr>`;
      return;
    }

    logsTableBody.innerHTML = logs.map(log => {
      const dateStr = new Date(log.timestamp * 1000).toLocaleString();
      const pred = (log.prediction || "unknown").toLowerCase();
      const conf = ((log.confidence || 0) * 100).toFixed(1);
      const probs = log.class_probabilities || {};
      const probStr = `${((probs.low||0)*100).toFixed(0)}% / ${((probs.medium||0)*100).toFixed(0)}% / ${((probs.high||0)*100).toFixed(0)}%`;

      return `
        <tr>
          <td>${dateStr}</td>
          <td><code>${log.filename || "image"}</code></td>
          <td><span class="severity-badge ${pred}" style="font-size:0.75rem; padding:4px 10px;">${pred.toUpperCase()}</span></td>
          <td><strong>${conf}%</strong></td>
          <td>${probStr}</td>
        </tr>
      `;
    }).join("");
  }

  // --- 8. Human-in-the-Loop Review Queue ---
  async function fetchReviewQueue() {
    reviewGrid.innerHTML = `<p class="table-empty">Loading review queue...</p>`;
    try {
      const res = await fetch("/api/review-queue");
      const items = await res.json();
      renderReviewQueue(items);
    } catch (err) {
      reviewGrid.innerHTML = `<p class="table-empty">Error loading review queue.</p>`;
    }
  }

  refreshReviewBtn.addEventListener("click", fetchReviewQueue);

  function renderReviewQueue(items) {
    if (!items || items.length === 0) {
      reviewGrid.innerHTML = `<p class="table-empty">No pending images to review. Upload some in Live Inference first.</p>`;
      return;
    }

    reviewGrid.innerHTML = items.map(item => {
      const pred = (item.prediction || "unknown").toLowerCase();
      const conf = ((item.confidence || 0) * 100).toFixed(1);
      return `
        <div class="review-card" data-saved-as="${item.saved_as}">
          <img src="${item.image_url}" alt="Incoming pothole image" class="review-card-img" />
          <div class="review-card-body">
            <div>
              <span class="severity-badge ${pred}" style="font-size:0.75rem; padding:4px 10px;">${pred.toUpperCase()}</span>
              <span class="review-card-conf">${conf}% confidence</span>
            </div>
            <p class="review-card-label">Confirm correct label:</p>
            <div class="review-card-actions">
              <button class="btn btn-outline review-btn" data-label="low">Low</button>
              <button class="btn btn-outline review-btn" data-label="medium">Medium</button>
              <button class="btn btn-outline review-btn" data-label="high">High</button>
            </div>
          </div>
        </div>
      `;
    }).join("");

    // Wire up each card's three buttons
    document.querySelectorAll(".review-btn").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        const card = e.target.closest(".review-card");
        const savedAs = card.dataset.savedAs;
        const label = e.target.dataset.label;
        await submitReview(savedAs, label, card);
      });
    });
  }

  async function submitReview(savedAs, label, cardEl) {
    try {
      const res = await fetch("/api/review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ saved_as: savedAs, confirmed_label: label })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      cardEl.remove(); // Instantly remove the reviewed card from view
    } catch (err) {
      alert(`Failed to submit review: ${err.message}`);
    }
  }

});
