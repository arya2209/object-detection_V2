// ==========================================
// OBJECT DETECTION CAMERA
// TensorFlow.js + COCO-SSD + Firebase
// ==========================================

// ==========================================
// FIREBASE
// ==========================================

import { initializeApp } from "https://www.gstatic.com/firebasejs/12.3.0/firebase-app.js";
import {
  getFirestore,
  collection,
  addDoc,
  getDocs,
  writeBatch,
  query,
  where,
  onSnapshot,
  serverTimestamp,
} from "https://www.gstatic.com/firebasejs/12.3.0/firebase-firestore.js";

const firebaseConfig = {
  apiKey: "AIzaSyC7hiFQ3HXza9gwjpzXV7Tm8xRsqe14p6k",
  authDomain: "expense-tracer-mahasiswa-c5cf8.firebaseapp.com",
  projectId: "expense-tracer-mahasiswa-c5cf8",
  storageBucket: "expense-tracer-mahasiswa-c5cf8.firebasestorage.app",
  messagingSenderId: "90640908734",
  appId: "1:90640908734:web:05f8de2843105240c5f9ef",
};

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
const detectionsCollection = collection(db, "expenses");

// Hanya dokumen dengan type "detection" yang dianggap history deteksi
const detectionsQuery = query(
  detectionsCollection,
  where("type", "==", "detection"),
);

// ==========================================
// ELEMENT HTML
// ==========================================

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const cameraBox = document.getElementById("cameraBox");

const loading = document.getElementById("loading");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const startCameraBtn = document.getElementById("startCameraBtn");

const cameraStatus = document.getElementById("cameraStatus");
const statusDot = document.getElementById("statusDot");

const cameraBtn = document.getElementById("cameraBtn");
const flashBtn = document.getElementById("flashBtn");
const switchBtn = document.getElementById("switchBtn");
const detectBtn = document.getElementById("detectBtn");
const personBtn = document.getElementById("personBtn");
const toastEl = document.getElementById("toast");

const detectedObject = document.getElementById("detectedObject");
const confidence = document.getElementById("confidence");
const lastDetection = document.getElementById("lastDetection");

const historyList = document.getElementById("historyList");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");

// ==========================================
// VARIABLE
// ==========================================

let model = null;
let stream = null;
let videoTrack = null;

let detectionOn = false; // tombol Deteksi
let personOnly = false; // tombol Person
let flashOn = false; // tombol Flash
let facing = "environment"; // "environment" = belakang, "user" = depan
let switching = false; // sedang berpindah kamera
let loopRunning = false;

// Pengaturan penyimpanan
const MIN_SCORE = 0.6; // minimal confidence 60%
const SAVE_COOLDOWN = 5000; // objek yang sama disimpan lagi setelah 5 detik
const MAX_HISTORY = 50; // jumlah history yang ditampilkan

let lastSavedClass = null;
let lastSavedTime = 0;

// ==========================================
// UI HELPER
// ==========================================

let toastTimer;

function toast(message) {
  toastEl.textContent = message;
  toastEl.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove("show"), 2500);
}

// Atur tampilan satu tombol kontrol
function setButton(btn, active, icon, state) {
  btn.classList.toggle("active", active);
  btn.querySelector("i").className = `fa-solid ${icon}`;
  btn.querySelector("small").textContent = state;
}

function updateUI() {
  const cameraActive = !!stream;

  setButton(
    cameraBtn,
    cameraActive,
    cameraActive ? "fa-video" : "fa-video-slash",
    cameraActive ? "On" : "Off",
  );

  setButton(
    flashBtn,
    flashOn,
    flashOn ? "fa-bolt" : "fa-bolt",
    flashOn ? "On" : "Off",
  );

  setButton(
    switchBtn,
    false,
    "fa-camera-rotate",
    facing === "user" ? "Depan" : "Belakang",
  );

  video.classList.toggle("mirror", facing === "user");

  setButton(
    detectBtn,
    detectionOn,
    detectionOn ? "fa-eye" : "fa-eye-slash",
    detectionOn ? "On" : "Off",
  );

  setButton(personBtn, personOnly, "fa-person", personOnly ? "Saja" : "Semua");

  flashBtn.disabled = !cameraActive;
  switchBtn.disabled = switching;
  detectBtn.disabled = !cameraActive;

  statusDot.classList.toggle("on", cameraActive);
  statusDot.classList.toggle("detecting", cameraActive && detectionOn);

  cameraStatus.textContent = !cameraActive
    ? "Kamera Mati"
    : detectionOn
      ? "Mendeteksi"
      : "Kamera Aktif";
}

function resetInfo() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  detectedObject.textContent = "-";
  confidence.textContent = "-";
}

// ==========================================
// LOAD COCO-SSD
// ==========================================

async function loadModel() {
  try {
    loading.style.display = "block";
    cameraPlaceholder.style.display = "none";

    model = await cocoSsd.load();

    loading.style.display = "none";
    cameraPlaceholder.style.display = "block";
  } catch (error) {
    console.error("Gagal memuat model:", error);

    loading.innerHTML = `<p>Gagal memuat model AI.</p>`;
  }
}

// ==========================================
// MENCARI KAMERA BELAKANG
// ==========================================

async function getBackCamera() {
  try {
    // Minta izin kamera terlebih dahulu
    const temporaryStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });

    const devices = await navigator.mediaDevices.enumerateDevices();

    temporaryStream.getTracks().forEach((track) => track.stop());

    const backCamera = devices
      .filter((device) => device.kind === "videoinput")
      .find((camera) => {
        const label = camera.label.toLowerCase();
        return (
          label.includes("back") ||
          label.includes("rear") ||
          label.includes("environment") ||
          label.includes("belakang")
        );
      });

    return backCamera ? backCamera.deviceId : null;
  } catch (error) {
    console.error("Gagal mendapatkan daftar kamera:", error);
    return null;
  }
}

// ==========================================
// START / STOP CAMERA
// ==========================================

async function startCamera() {
  if (!model) {
    toast("Model AI belum selesai dimuat.");
    return;
  }

  try {
    let videoConstraints;

    if (facing === "user") {
      videoConstraints = { facingMode: { ideal: "user" } };
    } else {
      const backCameraId = await getBackCamera();

      videoConstraints = backCameraId
        ? { deviceId: { exact: backCameraId } }
        : { facingMode: { ideal: "environment" } };
    }

    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        ...videoConstraints,
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
      audio: false,
    });

    videoTrack = stream.getVideoTracks()[0];

    video.srcObject = stream;
    video.style.display = "block";
    cameraPlaceholder.style.display = "none";

    await video.play();

    setCanvasSize();
    updateUI();

    if (detectionOn) detectLoop();
  } catch (error) {
    console.error("Gagal mengakses kamera:", error);

    stream = null;
    videoTrack = null;
    updateUI();

    toast("Kamera tidak dapat diakses. Periksa izin kamera.");
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }

  stream = null;
  videoTrack = null;
  flashOn = false;

  video.srcObject = null;
  video.style.display = "none";
  cameraPlaceholder.style.display = "block";
  cameraBox.style.aspectRatio = "";

  resetInfo();
  updateUI();
}

function toggleCamera() {
  if (stream) stopCamera();
  else startCamera();
}

async function switchCamera() {
  if (switching) return;

  const previous = facing;
  facing = facing === "user" ? "environment" : "user";

  // Kamera belum aktif: cukup simpan pilihan untuk nanti
  if (!stream) {
    updateUI();
    toast(
      facing === "user" ? "Kamera depan dipilih" : "Kamera belakang dipilih",
    );
    return;
  }

  switching = true;
  updateUI();

  // Matikan kamera lama (loop deteksi tetap hidup)
  stream.getTracks().forEach((track) => track.stop());
  flashOn = false;

  await startCamera();

  // Gagal (mis. perangkat tidak punya kamera depan): kembali ke kamera sebelumnya
  if (!stream) {
    facing = previous;
    await startCamera();
    toast("Kamera tersebut tidak tersedia di perangkat ini.");
  }

  switching = false;
  updateUI();
}
// ==========================================
// FLASH (TORCH)
// ==========================================

async function toggleFlash() {
  if (!videoTrack) return;

  const capabilities = videoTrack.getCapabilities
    ? videoTrack.getCapabilities()
    : {};

  if (!capabilities.torch) {
    toast("Flash tidak didukung di perangkat/browser ini.");
    return;
  }

  try {
    await videoTrack.applyConstraints({
      advanced: [{ torch: !flashOn }],
    });

    flashOn = !flashOn;
    updateUI();
  } catch (error) {
    console.error("Gagal mengubah flash:", error);
    toast("Gagal menyalakan flash.");
  }
}

// ==========================================
// TOGGLE DETEKSI & PERSON ONLY
// ==========================================

function toggleDetection() {
  if (!stream) return;

  detectionOn = !detectionOn;
  updateUI();

  if (detectionOn) detectLoop();
  else resetInfo();
}

function togglePersonOnly() {
  personOnly = !personOnly;
  updateUI();
  toast(personOnly ? "Hanya mendeteksi person" : "Mendeteksi semua objek");
}

// ==========================================
// SET CANVAS SIZE
// ==========================================

function setCanvasSize() {
  if (video.videoWidth === 0 || video.videoHeight === 0) {
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // Rasio kotak kamera mengikuti video agar bounding box pas
  cameraBox.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
}

// ==========================================
// OBJECT DETECTION
// ==========================================

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function detectLoop() {
  if (loopRunning) return;
  loopRunning = true;

  while (detectionOn && stream) {
    try {
      if (video.readyState >= 2) {
        let predictions = await model.detect(video);

        // Mungkin dimatikan saat model berjalan
        if (!detectionOn || !stream) break;

        if (personOnly) {
          predictions = predictions.filter((p) => p.class === "person");
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (predictions.length > 0) {
          const highest = predictions.reduce((best, p) =>
            p.score > best.score ? p : best,
          );

          showDetectionInfo(highest);
          saveDetection(highest);
          predictions.forEach(drawBoundingBox);
        } else {
          detectedObject.textContent = "-";
          confidence.textContent = "-";
        }
      }
    } catch (error) {
      console.error("Detection error:", error);
    }

    await sleep(200);
  }

  loopRunning = false;
  resetInfo();
}

// ==========================================
// DRAW BOUNDING BOX
// ==========================================

function drawBoundingBox(prediction) {
  const [bx, y, width, height] = prediction.bbox;

  // Kamera depan ditampilkan seperti cermin, jadi kotak ikut dibalik
  const x = facing === "user" ? canvas.width - bx - width : bx;
  const score = Math.round(prediction.score * 100);
  const label = `${prediction.class} ${score}%`;

  const color = "#d0b886";

  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, width, height);

  ctx.font = "600 16px system-ui, Arial";

  const textWidth = ctx.measureText(label).width;
  const labelHeight = 26;
  const labelY = Math.max(0, y - labelHeight);

  ctx.fillStyle = color;
  ctx.fillRect(x, labelY, textWidth + 14, labelHeight);

  ctx.fillStyle = "#17140e";
  ctx.fillText(label, x + 7, labelY + 18);
}

// ==========================================
// SHOW DETECTION INFO
// ==========================================

function showDetectionInfo(prediction) {
  const objectName = prediction.class;
  const score = Math.round(prediction.score * 100);

  detectedObject.textContent = objectName;
  confidence.textContent = `${score}%`;

  const time = new Date().toLocaleTimeString("id-ID", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  lastDetection.textContent = time;
}

// ==========================================
// SIMPAN DETECTION KE FIRESTORE
// ==========================================

async function saveDetection(prediction) {
  if (prediction.score < MIN_SCORE) return;

  const now = Date.now();

  // Hindari menyimpan objek yang sama terus-menerus
  // (deteksi berjalan tiap 200 ms)
  if (
    prediction.class === lastSavedClass &&
    now - lastSavedTime < SAVE_COOLDOWN
  ) {
    return;
  }

  lastSavedClass = prediction.class;
  lastSavedTime = now;

  try {
    await addDoc(detectionsCollection, {
      type: "detection",
      object: prediction.class,
      confidence: Math.round(prediction.score * 100),
      status: "Detected",
      createdAt: serverTimestamp(),
    });
  } catch (error) {
    console.error("Gagal menyimpan ke Firebase:", error);
  }
}

// ==========================================
// TAMPILKAN HISTORY (REALTIME)
// ==========================================

const EMPTY_HISTORY_HTML = `
    <tr id="emptyHistory">
        <td colspan="5">
            <div class="empty-history">
                <i class="fa-solid fa-clock-rotate-left"></i>
                <h3>Belum Ada History</h3>
                <p>Hasil deteksi objek akan muncul di sini.</p>
            </div>
        </td>
    </tr>
`;

function escapeHTML(text) {
  const div = document.createElement("div");
  div.textContent = text ?? "";
  return div.innerHTML;
}

function renderHistory(docs) {
  // Urutkan dari terbaru, ambil sejumlah MAX_HISTORY
  // "estimate" supaya data yang baru ditulis tetap punya waktu
  const items = docs
    .map((docSnap) => docSnap.data({ serverTimestamps: "estimate" }))
    .sort((a, b) => {
      const timeA = a.createdAt ? a.createdAt.toMillis() : 0;
      const timeB = b.createdAt ? b.createdAt.toMillis() : 0;
      return timeB - timeA;
    })
    .slice(0, MAX_HISTORY);

  if (items.length === 0) {
    historyList.innerHTML = EMPTY_HISTORY_HTML;
    return;
  }

  historyList.innerHTML = items
    .map((data, index) => {
      const time = data.createdAt
        ? data.createdAt.toDate().toLocaleString("id-ID", {
            day: "2-digit",
            month: "short",
            year: "numeric",
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
          })
        : "-";

      return `
            <tr>
                <td>${index + 1}</td>
                <td>
                    <div class="object-name">
                        <div class="object-icon">
                            <i class="fa-solid fa-cube"></i>
                        </div>
                        ${escapeHTML(data.object)}
                    </div>
                </td>
                <td class="confidence">${escapeHTML(String(data.confidence))}%</td>
                <td>${time}</td>
                <td><span class="status-badge">${escapeHTML(data.status)}</span></td>
            </tr>
        `;
    })
    .join("");
}

function listenHistory() {
  onSnapshot(
    detectionsQuery,
    (snapshot) => renderHistory(snapshot.docs),
    (error) => {
      console.error("Gagal membaca history:", error);
    },
  );
}

// ==========================================
// HAPUS HISTORY (HANYA DATA DETEKSI)
// ==========================================

async function clearHistory() {
  if (!confirm("Hapus semua history deteksi?")) return;

  try {
    const snapshot = await getDocs(detectionsQuery);

    // Batch Firestore maksimal 500 operasi
    let batch = writeBatch(db);
    let count = 0;

    for (const docSnap of snapshot.docs) {
      batch.delete(docSnap.ref);
      count++;

      if (count === 500) {
        await batch.commit();
        batch = writeBatch(db);
        count = 0;
      }
    }

    if (count > 0) await batch.commit();

    lastSavedClass = null;
  } catch (error) {
    console.error("Gagal menghapus history:", error);
    alert("Gagal menghapus history.");
  }
}

// ==========================================
// EVENT LISTENER
// ==========================================

startCameraBtn.addEventListener("click", startCamera);
cameraBtn.addEventListener("click", toggleCamera);
flashBtn.addEventListener("click", toggleFlash);
switchBtn.addEventListener("click", switchCamera);
detectBtn.addEventListener("click", toggleDetection);
personBtn.addEventListener("click", togglePersonOnly);

clearHistoryBtn.addEventListener("click", clearHistory);

// Matikan kamera saat tab disembunyikan agar hemat baterai
document.addEventListener("visibilitychange", () => {
  if (document.hidden && stream) stopCamera();
});

// ==========================================
// LOAD HISTORY & MODEL
// ==========================================

listenHistory();
loadModel();

updateUI();
