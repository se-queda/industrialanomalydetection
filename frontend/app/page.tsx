"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";

type ScanResult = {
  anomaly_score: number;
  is_anomaly: boolean;
  heatmap_url: string;
  inference_time_ms: number;
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewURL, setPreviewURL] = useState<string | null>(null);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [toast, setToast] = useState<{ message: string; type: "success" | "error" } | null>(null);

  // Configurable endpoints via env, with localhost fallbacks
  const EXPRESS_URL = useMemo(() => process.env.NEXT_PUBLIC_EXPRESS_URL ?? "http://localhost:3001", []);
  const FASTAPI_URL = useMemo(() => process.env.NEXT_PUBLIC_FASTAPI_URL ?? "http://localhost:8000", []);

  useEffect(() => {
    // Initialize theme from localStorage
    try {
      const saved = localStorage.getItem("theme");
      if (saved === "dark") setDarkMode(true);
    } catch {}

    const checkHealth = async () => {
      try {
        await axios.get(`${EXPRESS_URL}/api/health`);
        setHealth(true);
      } catch {
        setHealth(false);
      }
    };
    checkHealth();
    const timer = setInterval(checkHealth, 30000);
    return () => clearInterval(timer);
  }, [EXPRESS_URL]);

  const handleFileChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const selected = event.target.files?.[0] ?? null;
      setFile(selected);
      setResult(null);
      setError(null);
      if (previewURL) URL.revokeObjectURL(previewURL);
      setPreviewURL(selected ? URL.createObjectURL(selected) : null);
    },
    [previewURL]
  );

  const handleScan = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const formData = new FormData();
      // Accepts either 'image' or 'file' on the Express side
      formData.append("image", file);
      const res = await axios.post<ScanResult>(`${EXPRESS_URL}/api/scan`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
      setToast({ message: "Scan complete", type: "success" });
    } catch (err) {
      console.error(err);
      setError("Failed to process image.");
      setToast({ message: "Failed to process image", type: "error" });
    } finally {
      setLoading(false);
    }
  }, [file, EXPRESS_URL]);

  const handleReset = useCallback(() => {
    if (previewURL) URL.revokeObjectURL(previewURL);
    setFile(null);
    setPreviewURL(null);
    setResult(null);
    setError(null);
  }, [previewURL]);

  const toggleDarkMode = useCallback(() => setDarkMode((v) => !v), []);

  // Apply dark theme to <body> so the background covers full viewport
  useEffect(() => {
    if (typeof document !== "undefined") {
      document.body.classList.toggle("dark-theme", darkMode);
      try {
        localStorage.setItem("theme", darkMode ? "dark" : "light");
      } catch {}
    }
  }, [darkMode]);

  // Drag & drop handlers
  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);
  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);
  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const f = e.dataTransfer.files?.[0];
      if (f && f.type.startsWith("image/")) {
        if (previewURL) URL.revokeObjectURL(previewURL);
        setFile(f);
        setResult(null);
        setError(null);
        setPreviewURL(URL.createObjectURL(f));
        setToast({ message: `Loaded ${f.name}`, type: "success" });
      } else {
        setToast({ message: "Please drop an image file", type: "error" });
      }
    },
    [previewURL]
  );

  // Paste-from-clipboard support
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const it of items) {
        if (it.type.startsWith("image/")) {
          const blob = it.getAsFile();
          if (blob) {
            if (previewURL) URL.revokeObjectURL(previewURL);
            setFile(blob);
            setResult(null);
            setError(null);
            setPreviewURL(URL.createObjectURL(blob));
            setToast({ message: "Image pasted", type: "success" });
            break;
          }
        }
      }
    };
    document.addEventListener("paste", onPaste);
    return () => document.removeEventListener("paste", onPaste);
  }, [previewURL]);

  const containerClass = darkMode ? "container dark" : "container";
  const anomalyFlagClass = result?.is_anomaly ? "flag anomaly" : "flag normal";

  return (
    <div className={containerClass}>
      <header className="hero">
        <div className="logo">ReConPatch</div>
        <button className="toggle-dark top" onClick={toggleDarkMode}>
          {darkMode ? "Light" : "Dark"}
        </button>
      </header>
      <h1 className="title">Industrial Anomaly Scanner</h1>
      <p className="subtitle">Upload, drop, or paste an image to generate an anomaly heatmap.</p>

      <div className="controls">
        <label className="upload-label">
          Select Image
          <input type="file" accept="image/*" onChange={handleFileChange} />
        </label>
        {file && !result && (
          <button className="scan-button" onClick={handleScan} disabled={loading}>
            {loading ? "Scanning…" : "Scan Image"}
          </button>
        )}
        {result && (
          <button className="reset-button" onClick={handleReset}>
            Scan Another
          </button>
        )}
      </div>

      <div
        className={`image-container card ${isDragging ? "drop-highlight" : ""}`}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        {previewURL ? (
          <img src={previewURL} alt="Preview" className="image-preview" loading="lazy" />
        ) : (
          <div className="loading-overlay">Drop or paste an image to begin</div>
        )}

        {loading && (
          <div className="loading-overlay">
            <div className="spinner" />
          </div>
        )}

        {result?.heatmap_url && (
          <img
            src={`${FASTAPI_URL}${result.heatmap_url}`}
            alt="Heatmap"
            className="heatmap-overlay"
            loading="lazy"
          />
        )}
      </div>

      {result && (
        <div className="results">
          <div className={anomalyFlagClass}>{result.is_anomaly ? "Anomaly Detected" : "Normal"}</div>
          <div>Score: {result.anomaly_score.toFixed(4)}</div>
          <div>Inference: {result.inference_time_ms.toFixed(2)} ms</div>
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {toast && (
        <div className={`toast ${toast.type}`} onAnimationEnd={() => setToast(null)}>
          {toast.message}
        </div>
      )}

      <style jsx>{`
        .hero { display:flex; align-items:center; justify-content:space-between; max-width: 980px; margin: 0 auto 1rem; padding: 1rem; }
        .logo { font-weight: 700; letter-spacing: 0.2px; font-size: 1.05rem; opacity: 0.9; }
        .title { text-align: center; margin: 0.25rem 0 0.25rem; font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; }
        .subtitle { text-align: center; color: var(--muted); margin-bottom: 1.25rem; }
        .container { max-width: 980px; margin: 0 auto; padding: 1rem 1rem 2rem; min-height: 100vh; transition: background 0.3s, color 0.3s; }
        .container.dark {}
        .controls { display: flex; gap: 1rem; align-items: center; justify-content: center; flex-wrap: wrap; margin-bottom: 1rem; }
        .card { background: var(--card); border: 1px solid var(--border); box-shadow: var(--shadow); backdrop-filter: blur(8px); }
        .upload-label { display:inline-flex; align-items:center; gap:0.5rem; background: linear-gradient(135deg, var(--accent), #7c3aed); color:#fff; padding:0.65rem 1.05rem; border-radius:10px; cursor:pointer; transition: transform .15s ease, box-shadow .2s ease, opacity .3s ease; box-shadow: 0 10px 20px rgba(37,99,235,0.25); }
        .upload-label:hover { transform: translateY(-1px); box-shadow: 0 12px 24px rgba(37,99,235,0.30); }
        .upload-label:active { transform: translateY(0); }
        .upload-label input { display: none; }
        .scan-button, .reset-button, .toggle-dark { padding: .65rem 1rem; border: none; border-radius: 10px; cursor: pointer; font-size: .95rem; transition: transform .15s ease, box-shadow .2s ease, opacity .3s ease; }
        .scan-button { color:#fff; background: linear-gradient(135deg, var(--accent), #7c3aed); box-shadow: 0 10px 20px rgba(37,99,235,0.25); }
        .scan-button:hover:not([disabled]) { transform: translateY(-1px); box-shadow: 0 12px 24px rgba(37,99,235,0.35); }
        .scan-button[disabled] { opacity: .7; cursor: not-allowed; }
        .reset-button { background: var(--danger); color:#fff; }
        .reset-button:hover { transform: translateY(-1px); }
        .toggle-dark { background: var(--accent-2); color:#032b17; }
        .toggle-dark.top { background: transparent; border: 1px solid var(--border); color: var(--fg); }
        .toggle-dark:hover { transform: translateY(-1px); }
        .health { display:flex; align-items:center; justify-content:center; gap:.5rem; margin-bottom: 1rem; }
        .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; box-shadow: 0 0 0 6px rgba(0,0,0,0.04) inset; }
        .dot.ok { background: #22c55e; }
        .dot.err { background: #ef4444; }
        .image-container { position: relative; width: clamp(256px, 60vw, 400px); height: clamp(256px, 60vw, 400px); margin: 0 auto; border: 1px solid var(--border); border-radius: 16px; overflow: hidden; }
        .drop-highlight { outline: 3px dashed var(--accent); outline-offset: -12px; }
        .image-preview { width: 100%; height: 100%; object-fit: cover; display: block; }
        .heatmap-overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; opacity: 0.6; animation: fadeIn 600ms ease; pointer-events: none; }
        .loading-overlay { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; background: rgba(0,0,0,0.25); backdrop-filter: blur(2px); color: var(--fg); }
        .spinner { border: 4px solid rgba(255,255,255,0.2); border-top: 4px solid var(--accent); border-radius: 50%; width: 42px; height: 42px; animation: spin 0.9s linear infinite; }
        .results { margin-top: 1rem; text-align: center; }
        .flag { font-weight: 600; padding: 0.5rem 1rem; border-radius: 999px; margin-bottom: 0.5rem; display: inline-block; animation: fadeIn 400ms ease; box-shadow: 0 8px 16px rgba(0,0,0,0.08); }
        .flag.anomaly { background: var(--danger); color: white; }
        .flag.normal { background: var(--accent-2); color: #032b17; }
        .error { color: var(--danger); text-align: center; margin-top: 1rem; }
        .toast { position: fixed; right: 16px; bottom: 16px; padding: .75rem 1rem; border-radius: 10px; color:#fff; box-shadow: var(--shadow); animation: fadeIn 200ms ease, fadeOut 200ms ease 2400ms forwards; }
        .toast.success { background: #16a34a; }
        .toast.error { background: #ef4444; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes fadeOut { to { opacity: 0; transform: translateY(4px); } }
      `}</style>
    </div>
  );
}
