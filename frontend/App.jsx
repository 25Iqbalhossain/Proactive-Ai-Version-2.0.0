import React, { useEffect, useMemo, useState, useCallback } from 'react';

const SMART_API = '/smart-db-csv/api';
const CORE_API = '';
const TOKEN_KEY = 'proactive_ai_token';

const DB_OPTS = [
  ['mysql','MySQL / MariaDB'],['postgres','PostgreSQL'],
  ['mssql','SQL Server'],['mongodb','MongoDB'],['sqlite','SQLite'],
];

const PORT_MAP = { postgres:'5432', mysql:'3306', mssql:'1433', mongodb:'27017', sqlite:'' };

const DB_ICONS = {
  mysql: '🐬', postgres: '🐘', mssql: '🪟', mongodb: '🍃', sqlite: '📦'
};

// ── Global Styles ─────────────────────────────────────────────────────────────
const css = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

  :root {
    --bg: #05080f;
    --surface: #0c1120;
    --surface2: #111827;
    --surface3: #1a2235;
    --border: rgba(99,130,195,0.15);
    --border-hi: rgba(99,130,195,0.35);
    --text: #e8eeff;
    --muted: #6b7fa3;
    --accent: #4f7cff;
    --accent2: #7c3aed;
    --green: #34d399;
    --yellow: #fbbf24;
    --red: #f87171;
    --cyan: #22d3ee;
    --glow: rgba(79,124,255,0.15);
    --glow2: rgba(124,58,237,0.15);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  html { scroll-behavior: smooth; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', system-ui, sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Animated background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
      radial-gradient(ellipse 80% 50% at 20% 10%, rgba(79,124,255,0.08) 0%, transparent 60%),
      radial-gradient(ellipse 60% 40% at 80% 80%, rgba(124,58,237,0.07) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
  }

  .app-wrap {
    position: relative;
    z-index: 1;
    max-width: 920px;
    margin: 0 auto;
    padding: 32px 20px 80px;
  }

  /* ── Header ── */
  .header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 40px;
    flex-wrap: wrap;
  }

  .logo-area { display: flex; flex-direction: column; gap: 6px; }

  .logo-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 4px 12px;
    background: rgba(79,124,255,0.1);
    border: 1px solid rgba(79,124,255,0.25);
    border-radius: 100px;
    font-size: 24px;
    font-weight: 1000;
    color: var(--accent);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    width: fit-content;
  }

  .logo-tag .dot {
    width: 6px; height: 6px;
    background: var(--accent);
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
  }

  .header h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(22px, 4vw, 30px);
    font-weight: 500;
    line-height: 1.1;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #e8eeff 0%, #a5b4fc 60%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .header p {
    font-size: 13px;
    color: var(--muted);
    line-height: 1.6;
    max-width: 500px;
  }

  /* ── Stepper ── */
  .stepper {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 32px;
    overflow-x: auto;
    padding-bottom: 4px;
    scrollbar-width: none;
  }
  .stepper::-webkit-scrollbar { display: none; }

  .step-item {
    display: flex;
    align-items: center;
    gap: 0;
    flex-shrink: 0;
  }

  .step-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 100px;
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    cursor: pointer;
    transition: all 0.25s ease;
    white-space: nowrap;
    letter-spacing: 0.02em;
  }

  .step-btn:hover:not(.active) {
    border-color: var(--border-hi);
    color: var(--text);
    background: rgba(255,255,255,0.03);
  }

  .step-btn.active {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-color: transparent;
    color: #fff;
    box-shadow: 0 0 20px rgba(79,124,255,0.35);
  }

  .step-btn.done {
    border-color: rgba(52,211,153,0.3);
    color: var(--green);
    background: rgba(52,211,153,0.06);
  }

  .step-num {
    width: 20px; height: 20px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px;
    font-weight: 700;
    background: rgba(255,255,255,0.1);
    flex-shrink: 0;
  }

  .step-btn.active .step-num { background: rgba(255,255,255,0.25); }
  .step-btn.done .step-num { background: rgba(52,211,153,0.2); }

  .step-connector {
    width: 24px;
    height: 1px;
    background: var(--border);
    flex-shrink: 0;
  }

  /* ── Panel / Card ── */
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 24px;
    position: relative;
    overflow: hidden;
    animation: fadeUp 0.35s ease;
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .panel::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(79,124,255,0.04) 0%, transparent 60%);
    pointer-events: none;
  }

  .panel + .panel { margin-top: 16px; }

  .panel-header {
    margin-bottom: 20px;
  }

  .panel-header h2 {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 6px;
    letter-spacing: -0.01em;
  }

  .panel-header p {
    font-size: 13px;
    color: var(--muted);
    line-height: 1.55;
  }

  /* ── Form elements ── */
  .form-stack { display: flex; flex-direction: column; gap: 16px; }

  .form-group { display: flex; flex-direction: column; gap: 6px; }

  .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .form-row3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }

  label {
    font-size: 11px;
    font-weight: 600;
    color: var(--muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  input, select, textarea {
    width: 100%;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 12px;
    padding: 11px 14px;
    font-size: 14px;
    font-family: 'DM Sans', sans-serif;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
    -webkit-appearance: none;
  }

  input:focus, select:focus, textarea:focus {
    border-color: rgba(79,124,255,0.5);
    box-shadow: 0 0 0 3px rgba(79,124,255,0.1);
  }

  select {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%236b7fa3' d='M6 8L1 3h10z'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 12px center;
    padding-right: 36px;
    cursor: pointer;
  }

  textarea { min-height: 90px; resize: vertical; line-height: 1.5; }

  input[type="checkbox"] {
    width: 18px; height: 18px;
    border-radius: 5px;
    cursor: pointer;
    accent-color: var(--accent);
    flex-shrink: 0;
  }

  /* ── Buttons ── */
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 11px 20px;
    border-radius: 12px;
    font-size: 14px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    cursor: pointer;
    border: none;
    transition: all 0.2s ease;
    white-space: nowrap;
    letter-spacing: 0.01em;
  }

  .btn-primary {
    background: linear-gradient(135deg, var(--accent), #6366f1);
    color: #fff;
    box-shadow: 0 4px 16px rgba(79,124,255,0.3);
  }

  .btn-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 24px rgba(79,124,255,0.45);
  }

  .btn-secondary {
    background: var(--surface2);
    color: var(--text);
    border: 1px solid var(--border);
  }

  .btn-secondary:hover {
    border-color: var(--border-hi);
    background: var(--surface3);
  }

  .btn-ghost {
    background: transparent;
    color: var(--muted);
    border: 1px solid var(--border);
  }

  .btn-ghost:hover { color: var(--text); border-color: var(--border-hi); }

  .btn-danger {
    background: rgba(248,113,113,0.1);
    color: var(--red);
    border: 1px solid rgba(248,113,113,0.2);
  }

  .btn-danger:hover {
    background: rgba(248,113,113,0.18);
    border-color: rgba(248,113,113,0.4);
  }

  .btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
    transform: none !important;
    box-shadow: none !important;
  }

  .btn-sm { padding: 6px 12px; font-size: 12px; border-radius: 8px; }

  .actions {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    align-items: center;
    margin-top: 4px;
  }

  /* ── Badges ── */
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 100px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.04em;
    border: 1px solid var(--border);
    color: var(--muted);
    background: rgba(255,255,255,0.03);
    white-space: nowrap;
  }

  .badge-green { color: var(--green); border-color: rgba(52,211,153,0.25); background: rgba(52,211,153,0.08); }
  .badge-yellow { color: var(--yellow); border-color: rgba(251,191,36,0.25); background: rgba(251,191,36,0.08); }
  .badge-red { color: var(--red); border-color: rgba(248,113,113,0.25); background: rgba(248,113,113,0.08); }
  .badge-blue { color: var(--accent); border-color: rgba(79,124,255,0.25); background: rgba(79,124,255,0.08); }
  .badge-purple { color: #a78bfa; border-color: rgba(167,139,250,0.25); background: rgba(167,139,250,0.08); }

  /* ── Connection cards ── */
  .conn-card {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    padding: 14px 16px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
    transition: border-color 0.2s, background 0.2s;
  }

  .conn-card:hover { border-color: var(--border-hi); }
  .conn-card.selected { border-color: rgba(79,124,255,0.35); background: rgba(79,124,255,0.06); }

  .conn-info { display: flex; align-items: center; gap: 12px; }

  .conn-icon {
    width: 36px; height: 36px;
    border-radius: 10px;
    background: var(--surface3);
    border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
  }

  .conn-name { font-weight: 600; font-size: 14px; }
  .conn-detail { font-size: 12px; color: var(--muted); margin-top: 2px; }

  .conn-actions { display: flex; gap: 8px; align-items: center; }

  /* ── Source mode cards ── */
  .source-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 8px;
  }

  .source-card {
    border: 2px solid var(--border);
    border-radius: 18px;
    padding: 28px 24px;
    cursor: pointer;
    text-align: center;
    transition: all 0.25s ease;
    background: var(--surface2);
    position: relative;
    overflow: hidden;
  }

  .source-card::before {
    content: '';
    position: absolute;
    inset: 0;
    opacity: 0;
    transition: opacity 0.25s;
  }

  .source-card.csv::before {
    background: radial-gradient(ellipse at 50% 0%, rgba(79,124,255,0.12) 0%, transparent 70%);
  }

  .source-card.db::before {
    background: radial-gradient(ellipse at 50% 0%, rgba(124,58,237,0.12) 0%, transparent 70%);
  }

  .source-card:hover::before, .source-card.sel::before { opacity: 1; }

  .source-card:hover, .source-card.sel {
    border-color: var(--accent);
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(79,124,255,0.2);
  }

  .source-card.db:hover, .source-card.db.sel {
    border-color: #7c3aed;
    box-shadow: 0 12px 40px rgba(124,58,237,0.2);
  }

  .source-icon {
    font-size: 40px;
    display: block;
    margin-bottom: 12px;
    filter: drop-shadow(0 4px 12px rgba(0,0,0,0.3));
  }

  .source-card h3 {
    font-family: 'Syne', sans-serif;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 6px;
    letter-spacing: -0.01em;
  }

  .source-card p {
    font-size: 13px;
    color: var(--muted);
    line-height: 1.5;
  }

  /* ── Log / Code ── */
  .log-box {
    background: #070b14;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    overflow: auto;
    max-height: 300px;
    white-space: pre-wrap;
    font-family: 'DM Mono', ui-monospace, monospace;
    font-size: 11px;
    color: #7b93c4;
    line-height: 1.6;
  }

  /* ── Messages ── */
  .msg-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 12px 16px;
    border-radius: 12px;
    font-size: 13px;
    margin-bottom: 16px;
    border: 1px solid var(--border);
    background: var(--surface2);
    animation: slideDown 0.25s ease;
  }

  @keyframes slideDown {
    from { opacity: 0; transform: translateY(-8px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .msg-bar.err { border-color: rgba(248,113,113,0.3); background: rgba(248,113,113,0.07); color: var(--red); }
  .msg-bar.ok { border-color: rgba(52,211,153,0.3); background: rgba(52,211,153,0.07); color: var(--green); }

  /* ── Progress / Loading ── */
  .loading-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
    padding: 40px 20px;
    text-align: center;
  }

  .spinner-ring {
    width: 60px; height: 60px;
    border-radius: 50%;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    animation: spin 0.8s linear infinite;
    position: relative;
  }

  .spinner-ring::after {
    content: '';
    position: absolute;
    inset: 4px;
    border-radius: 50%;
    border: 2px solid transparent;
    border-top-color: rgba(124,58,237,0.5);
    animation: spin 1.2s linear infinite reverse;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  .progress-track {
    width: 100%;
    height: 6px;
    background: var(--surface3);
    border-radius: 100px;
    overflow: hidden;
    margin-top: 8px;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 100px;
    transition: width 0.5s ease;
    position: relative;
  }

  .progress-fill::after {
    content: '';
    position: absolute;
    right: 0; top: 0; bottom: 0;
    width: 30px;
    background: rgba(255,255,255,0.4);
    filter: blur(6px);
    animation: shimmer 1.5s ease-in-out infinite;
  }

  @keyframes shimmer {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 1; }
  }

  /* ── Segmented control ── */
  .seg-control {
    display: inline-flex;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
  }

  .seg-btn {
    padding: 7px 14px;
    border-radius: 9px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    border: none;
    background: transparent;
    color: var(--muted);
    transition: all 0.2s;
    font-family: 'DM Sans', sans-serif;
  }

  .seg-btn.active {
    background: var(--surface3);
    color: var(--text);
    box-shadow: 0 1px 4px rgba(0,0,0,0.3);
  }

  /* ── Recommendation card ── */
  .rec-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
    transition: border-color 0.2s;
  }

  .rec-card:hover { border-color: var(--border-hi); }

  .rec-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 8px;
    flex-wrap: wrap;
  }

  .rec-card-rank {
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 700;
  }

  /* ── Model card ── */
  .model-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
    transition: all 0.2s;
  }

  .model-card:hover { border-color: var(--border-hi); }

  .model-card-header {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 8px;
    flex-wrap: wrap;
  }

  .model-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 14px;
  }

  .model-note { font-size: 12px; color: var(--muted); line-height: 1.5; margin-top: 6px; }

  /* ── Divider ── */
  .divider {
    height: 1px;
    background: var(--border);
    margin: 20px 0;
  }

  /* ── Schema viewer ── */
  .schema-wrap { margin-top: 12px; }

  .schema-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
  }

  /* ── Cards grid ── */
  .cards-stack { display: flex; flex-direction: column; gap: 10px; }
  .mini-grid { display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); }

  /* ── Empty state ── */
  .empty-state {
    text-align: center;
    padding: 32px 20px;
    color: var(--muted);
    font-size: 13px;
  }

  .empty-icon { font-size: 36px; margin-bottom: 10px; opacity: 0.5; }

  /* ── Status inline ── */
  .status-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 12px;
    align-items: center;
  }

  /* ── Responsive ── */
  @media(max-width: 640px) {
    .form-row, .form-row3, .source-grid { grid-template-columns: 1fr; }
    .stepper { gap: 0; }
    .step-connector { width: 12px; }
    .step-btn { padding: 7px 10px; font-size: 11px; }
  }

  /* ── Tab indicator for build modes ── */
  .build-modes {
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }

  .mode-tab {
    padding: 7px 14px;
    border-radius: 10px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--muted);
    transition: all 0.2s;
    font-family: 'DM Sans', sans-serif;
  }

  .mode-tab.active {
    border-color: rgba(79,124,255,0.4);
    color: var(--accent);
    background: rgba(79,124,255,0.08);
  }

  .mode-tab:hover:not(.active) {
    border-color: var(--border-hi);
    color: var(--text);
  }

  /* ── Auto-advance banner ── */
  .advance-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    padding: 14px 18px;
    background: rgba(52,211,153,0.07);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 14px;
    font-size: 13px;
    color: var(--green);
    margin-top: 16px;
    animation: fadeUp 0.4s ease;
  }

  .advance-icon { font-size: 20px; flex-shrink: 0; }

  /* ── Glow accents ── */
  .glow-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    animation: pulse 2s infinite;
  }

  .glow-dot.green { background: var(--green); box-shadow: 0 0 8px var(--green); }
  .glow-dot.blue { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
  .glow-dot.yellow { background: var(--yellow); box-shadow: 0 0 8px var(--yellow); }
  .glow-dot.red { background: var(--red); box-shadow: 0 0 8px var(--red); }
`;

// ── Utilities ─────────────────────────────────────────────────────────────────

async function jfetch(url, opts = {}) {
  const res = await fetch(url, opts);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  const detail = data && (data.detail || data.message || data.error);
  const message = Array.isArray(detail)
    ? detail.map(item => {
        const path = Array.isArray(item?.loc) ? item.loc.filter(p => p !== 'body').join('.') : '';
        return [path, item?.msg].filter(Boolean).join(': ');
      }).join('; ')
    : detail;
  if (!res.ok) throw new Error(message || text || `HTTP ${res.status}`);
  return data;
}

function norm(c) {
  return {
    ...c,
    name: c.name || (c.cred && c.cred.name) || c.id,
    db_type: c.db_type || (c.cred && c.cred.db_type) || '?',
    database: c.database || (c.cred && c.cred.database) || '',
  };
}

function buildConnectionPayload(form) {
  const payload = { db_type: form.db_type, name: (form.name || '').trim() };
  const add = (key, value) => { if (value !== '' && value != null) payload[key] = value; };
  const database = (form.database || '').trim();
  const host = (form.host || '').trim();
  const username = (form.username || '').trim();
  const password = form.password || '';
  const filepath = (form.filepath || '').trim();
  const uri = (form.uri || '').trim();
  const port = (form.port || '').trim();
  add('database', database);
  if (form.db_type === 'sqlite') { add('filepath', filepath); return payload; }
  if (!(form.db_type === 'mongodb' && uri)) {
    add('host', host);
    add('port', port ? Number(port) : null);
  }
  add('username', username);
  add('password', password);
  add('uri', uri);
  return payload;
}

function validateConnectionPayload(payload) {
  const needsHost = ['mysql', 'postgres', 'mssql'].includes(payload.db_type)
    || (payload.db_type === 'mongodb' && !payload.uri);
  if (needsHost && !payload.host) throw new Error(`Host is required for ${payload.db_type}.`);
}

function isTerminal(status) { return ['done', 'failed', 'error'].includes(status); }
function num(v, d = 4) { const n = Number(v); return Number.isFinite(n) ? n.toFixed(d) : 'n/a'; }
function pct(v, d = 2) { const n = Number(v); return Number.isFinite(n) ? `${n.toFixed(d)}%` : 'n/a'; }
function recText(w) { if (!w) return ''; if (typeof w === 'string') return w; return w.message || w.detail || w.error || JSON.stringify(w); }

function uniqueModels(rows = []) {
  const seen = new Set();
  return rows.filter(row => { const k = row?.model_id || row?.algorithm; if (!k || seen.has(k)) return false; seen.add(k); return true; });
}

// ── UI Atoms ──────────────────────────────────────────────────────────────────

function Msg({ msg, clear }) {
  if (!msg) return null;
  const isErr = msg.toLowerCase().includes('error') || msg.toLowerCase().includes('fail') || msg.toLowerCase().includes('required');
  return (
    <div className={`msg-bar ${isErr ? 'err' : 'ok'}`}>
      <div style={{display:'flex',alignItems:'center',gap:10}}>
        <span>{isErr ? '⚠' : '✓'}</span>
        <span>{msg}</span>
      </div>
      <button className="btn btn-ghost btn-sm" onClick={clear}>✕</button>
    </div>
  );
}

function LoadingSpinner({ label, progress }) {
  return (
    <div className="loading-wrapper">
      <div className="spinner-ring" />
      <div>
        <div style={{fontWeight:600,marginBottom:4}}>{label || 'Processing…'}</div>
        {progress != null && (
          <>
            <div style={{fontSize:12,color:'var(--muted)',marginBottom:8}}>{progress}% complete</div>
            <div className="progress-track" style={{width:200}}>
              <div className="progress-fill" style={{width:`${progress}%`}} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function StatusDot({ status }) {
  const map = { done:'green', pending:'yellow', running:'blue', failed:'red', error:'red' };
  return <span className={`glow-dot ${map[status]||'blue'}`} />;
}

// ── JobStatus ─────────────────────────────────────────────────────────────────

function JobStatus({ data, label }) {
  if (!data) return null;
  const isRunning = !isTerminal(data.status);
  const training = data.result || null;
  const selectionPolicy = training?.model_selection_policy || null;
  const optunaPolicy = training?.optuna_policy || null;
  const selectedModels = [];

  const badgeClass = data.status === 'done' ? 'badge-green' : (data.status === 'failed' || data.status === 'error') ? 'badge-red' : 'badge-yellow';

  return (
    <div style={{marginTop:16}}>
      {isRunning && <LoadingSpinner label={`${label} in progress…`} progress={data.progress} />}
      <div className="status-row">
        <span className={`badge ${badgeClass}`}><StatusDot status={data.status} /> {label} — {data.status}</span>
        {data.progress != null && !isTerminal(data.status) && <span className="badge">{data.progress}%</span>}
        {data.row_count && <span className="badge badge-blue">Rows: {data.row_count}</span>}
        {data.result?.best_algorithm && <span className="badge badge-green">Best: {data.result.best_algorithm}</span>}
      </div>

      {training && (
        <div style={{display:'flex',flexDirection:'column',gap:12,marginBottom:12}}>
          {optunaPolicy?.summary && (
            <div className="panel" style={{padding:'14px 16px'}}>
              <div style={{fontSize:12,fontWeight:700,color:'var(--muted)',marginBottom:4,textTransform:'uppercase',letterSpacing:'0.06em'}}>Optuna policy</div>
              <div style={{fontSize:13}}>{optunaPolicy.summary}</div>
              {optunaPolicy.top_k_definition && <div style={{fontSize:12,color:'var(--muted)',marginTop:6}}>{optunaPolicy.top_k_definition}</div>}
            </div>
          )}
          {selectionPolicy?.reason && (
            <div className="panel" style={{padding:'14px 16px'}}>
              <div style={{fontSize:12,fontWeight:700,color:'var(--muted)',marginBottom:4,textTransform:'uppercase',letterSpacing:'0.06em'}}>{selectionPolicy.display_title || 'Model decision'}</div>
              <div style={{fontSize:13}}>{selectionPolicy.reason}</div>
            </div>
          )}
          {selectedModels.length > 0 && (
            <div className="cards-stack">
              {selectedModels.map(model => (
                <div key={model.model_id || model.algorithm} className="model-card">
                  <div className="model-card-header">
                    <div className="model-title">#{model.rank} {model.algorithm}</div>
                    <span className="badge badge-green">Score {Number(model.selection_score_pct||0).toFixed(2)}%</span>
                  </div>
                  <div className="model-note">
                    {(model.metric_name||'Metric')} {model.metric_value != null ? Number(model.metric_value).toFixed(4) : 'n/a'} · Composite {model.composite_score != null ? Number(model.composite_score).toFixed(4) : 'n/a'}
                  </div>
                  {model.summary && <div style={{fontSize:13,lineHeight:1.5,marginTop:8,color:'var(--text)'}}>{model.summary}</div>}
                  {model.decision_note && <div style={{fontSize:12,color:'var(--muted)',marginTop:6}}>{model.decision_note}</div>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      {data.error && <div className="panel" style={{padding:'14px 16px',borderColor:'rgba(248,113,113,0.3)'}}><strong style={{color:'var(--red)'}}>Error:</strong> {data.error}</div>}
      <div className="log-box">{JSON.stringify(data.plan || data.steps || data, null, 2)}</div>
    </div>
  );
}

// ── PageMode ──────────────────────────────────────────────────────────────────

function PageMode({ onSelect }) {
  const [hover, setHover] = useState('');
  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Choose data source</h2>
        <p>How would you like to supply training data for your recommendation models?</p>
      </div>
      <div className="source-grid">
        {[
          ['csv','📄','Upload CSV','Directly upload an existing CSV to train recommendation models instantly.',false],
          ['db','🗄️','Live Database','Connect to a DB, build a smart dataset with AI, then train models.',true],
        ].map(([k,icon,title,desc,isDb]) => (
          <div key={k} className={`source-card ${k} ${hover===k?'sel':''}`}
            onMouseEnter={()=>setHover(k)} onMouseLeave={()=>setHover('')}
            onClick={()=>onSelect(k)}>
            <span className="source-icon">{icon}</span>
            <h3>{title}</h3>
            <p>{desc}</p>
            <div style={{marginTop:16}}>
              <span className="badge" style={{fontSize:11}}>
                {k==='csv' ? '⚡ Quick start' : '🔗 Multi-source'}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── PageConnections ───────────────────────────────────────────────────────────

function PageConnections({ connections, selected, schemas, onAdded, onRemove, onToggle, onSchema, onAdvance, setMsg }) {
  const blank = { db_type:'postgres', name:'', host:'', port:'5432', database:'', username:'', password:'', filepath:'', uri:'' };
  const [form, setForm] = useState(blank);
  const set = (k,v) => setForm(p=>({...p,[k]:v}));
  const [saving, setSaving] = useState(false);
  const canAdvance = selected.length > 0;

  function changeType(t) { setForm(p=>({...p, db_type:t, port: PORT_MAP[t]||''})); }

  async function save() {
    setSaving(true);
    try {
      const payload = buildConnectionPayload(form);
      validateConnectionPayload(payload);
      const tested = await jfetch(`${SMART_API}/connections/test`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
      if (tested?.status && tested.status !== 'connected') throw new Error(tested.message || `Connection test failed.`);
      const res = await jfetch(`${SMART_API}/connections`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
      setMsg(`✓ Connected: ${res.name}`);
      setForm(blank);
      onAdded(res.id);
    } catch(e) { setMsg(e.message); }
    finally { setSaving(false); }
  }

  const isMongo = form.db_type === 'mongodb';
  const isSqlite = form.db_type === 'sqlite';

  return (
    <div style={{display:'flex',flexDirection:'column',gap:16}}>
      <div className="panel">
        <div className="panel-header">
          <h2>Add connection</h2>
          <p>Credentials are stored securely on the backend. New connections are auto-selected.</p>
        </div>
        <div className="form-stack">
          <div className="form-group">
            <label>Database type</label>
            <select value={form.db_type} onChange={e=>changeType(e.target.value)}>
              {DB_OPTS.map(([v,l])=><option key={v} value={v}>{DB_ICONS[v]||''} {l}</option>)}
            </select>
          </div>
          <div className="form-row">
            <div className="form-group"><label>Connection name</label><input value={form.name} onChange={e=>set('name',e.target.value)} placeholder="e.g. sales-db" /></div>
            <div className="form-group"><label>Database</label><input value={form.database} onChange={e=>set('database',e.target.value)} placeholder="e.g. analytics" /></div>
          </div>
          {!isSqlite && (
            <div className="form-row3">
              <div className="form-group"><label>Host</label><input value={form.host} onChange={e=>set('host',e.target.value)} placeholder="localhost" /></div>
              <div className="form-group"><label>Port</label><input value={form.port} onChange={e=>set('port',e.target.value)} /></div>
              <div className="form-group"><label>Username</label><input value={form.username} onChange={e=>set('username',e.target.value)} /></div>
            </div>
          )}
          {isSqlite
            ? <div className="form-group"><label>SQLite file path</label><input value={form.filepath} onChange={e=>set('filepath',e.target.value)} placeholder="/data/db.sqlite" /></div>
            : <div className="form-group"><label>Password</label><input type="password" value={form.password} onChange={e=>set('password',e.target.value)} placeholder="••••••••" /></div>}
          {isMongo && <div className="form-group"><label>MongoDB URI (optional)</label><input value={form.uri} onChange={e=>set('uri',e.target.value)} placeholder="mongodb://user:pass@host:27017/db" /></div>}
          <div className="actions">
            <button className="btn btn-primary" onClick={save} disabled={saving}>
              {saving ? '⏳ Testing…' : '⚡ Save & test connection'}
            </button>
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <h2>Saved connections</h2>
          <p>Check the connections you want to include in the build step.</p>
        </div>
        <div className="cards-stack">
          {connections.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon">🔌</div>
              <div>No connections yet — add one above</div>
            </div>
          )}
          {connections.map(c=>(
            <div key={c.id} className={`conn-card ${selected.includes(c.id)?'selected':''}`}>
              <div className="conn-info">
                <input type="checkbox" checked={selected.includes(c.id)}
                  onChange={e=>{ onToggle(c.id,e.target.checked); if(e.target.checked) onSchema(c.id); }} />
                <div className="conn-icon">{DB_ICONS[c.db_type]||'🗄️'}</div>
                <div>
                  <div className="conn-name">{c.name} <span className="badge badge-blue" style={{marginLeft:4,fontSize:10}}>{c.db_type}</span></div>
                  <div className="conn-detail">{c.database||'—'}</div>
                </div>
              </div>
              <div className="conn-actions">
                <button className="btn btn-secondary btn-sm" onClick={()=>onSchema(c.id)}>Schema</button>
                <button className="btn btn-danger btn-sm" onClick={()=>onRemove(c.id)}>Remove</button>
              </div>
            </div>
          ))}
        </div>
        {Object.entries(schemas).map(([id,s])=>(
          <div key={id} className="schema-wrap">
            <div className="schema-label">{connections.find(c=>c.id===id)?.name||id} — schema</div>
            <div className="log-box">{JSON.stringify((s.tables||[]).slice(0,8).map(t=>({table:t.table_name,cols:(t.columns||[]).map(c=>`${c.name}:${c.data_type}`)})),null,2)}</div>
          </div>
        ))}
        <div className="actions" style={{justifyContent:'space-between',marginTop:16}}>
          <div style={{fontSize:13,color:'var(--muted)'}}>
            {canAdvance
              ? `${selected.length} connection${selected.length === 1 ? '' : 's'} selected. Continue to Build.`
              : 'Select at least one connection to continue.'}
          </div>
          <button className="btn btn-primary" onClick={onAdvance} disabled={!canAdvance}>Next</button>
        </div>
      </div>
    </div>
  );
}

// ── PageBuild ─────────────────────────────────────────────────────────────────

function PageBuild({ selected, schemas, onBuilt, setMsg }) {
  const [form, setForm] = useState({
    mode:'query',
    rec_system_type:'hybrid',
    output_format:'csv',
    max_rows_per_table:50000,
    target_description:'',
    query_text:'',
    llm_prompt:'',
    manual_config:{ tables:'', relationships:'', target_field:'', label_field:'', notes:'' },
  });
  const [job, setJob] = useState(null);
  const [jobData, setJobData] = useState(null);
  const [autoLoadedJobId, setAutoLoadedJobId] = useState('');
  const [autoDownloadedJobId, setAutoDownloadedJobId] = useState('');
  const set = (k,v) => setForm(p=>({...p,[k]:v}));
  const setManual = (k,v) => setForm(p=>({...p,manual_config:{...p.manual_config,[k]:v}}));

  const availableTables = useMemo(
    ()=>selected.flatMap(id=>(schemas[id]?.tables||[]).map(t=>t.full_name||[t.schema_name,t.table_name].filter(Boolean).join('.'))),
    [schemas, selected]
  );

  useEffect(()=>{
    if (!job) return;
    const t = setInterval(async()=>{
      try {
        const d = await jfetch(`${SMART_API}/jobs/${job}`);
        setJobData(d);
        if (isTerminal(d.status)) clearInterval(t);
      } catch(e){ setMsg(e.message); clearInterval(t); }
    }, 2000);
    return ()=>clearInterval(t);
  },[job]);

  useEffect(()=>{
    if (!job || jobData?.status !== 'done' || autoDownloadedJobId === job) return;
    setAutoDownloadedJobId(job);
    download(true);
  },[autoDownloadedJobId, job, jobData?.status]);

  useEffect(()=>{
    if (!job || jobData?.status !== 'done' || form.output_format !== 'csv' || autoLoadedJobId === job) return;
    setAutoLoadedJobId(job);
    loadForTraining(true);
  },[autoLoadedJobId, form.output_format, job, jobData?.status]);

  async function start() {
    try {
      const mode = form.mode || 'query';
      const payload = {
        connection_ids:selected,
        mode,
        rec_system_type:form.rec_system_type,
        output_format:form.output_format,
        max_rows_per_table:Number(form.max_rows_per_table),
      };
      if (mode==='query') {
        const qt = (form.query_text || form.target_description || '').trim();
        if (!qt) throw new Error('Query mode requires a query or prompt.');
        payload.query_text = qt;
        payload.target_description = qt;
      } else if (mode==='llm') {
        const lp = (form.llm_prompt || '').trim();
        if (!lp) throw new Error('LLM Build mode requires instructions.');
        payload.llm_prompt = lp;
        payload.target_description = lp;
      } else {
        const mc = Object.fromEntries(Object.entries(form.manual_config||{}).map(([k,v])=>[k,(v||'').trim()]));
        if (!Object.values(mc).some(Boolean)) throw new Error('Manual mode requires at least one configuration field.');
        payload.manual_config = mc;
      }
      const res = await jfetch(`${SMART_API}/build`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
      setAutoLoadedJobId('');
      setAutoDownloadedJobId('');
      setJob(res.job_id);
      setJobData({status:'pending',progress:0});
      setMsg('Dataset build started.');
    } catch(e){ setMsg(e.message); }
  }

  async function loadForTraining(autoAdvance = false) {
    try {
      const res = await fetch(`${SMART_API}/jobs/${job}/download?output_format=csv`);
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      if (autoAdvance) setMsg('Dataset ready — advancing to Train. ✓');
      onBuilt(new File([blob],'built_dataset.csv',{type:'text/csv'}));
    } catch(e){ setMsg(e.message); }
  }

  async function download(autoTriggered = false) {
    try {
      const fmt = form.output_format||'csv';
      const res = await fetch(`${SMART_API}/jobs/${job}/download?output_format=${fmt}`);
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      Object.assign(document.createElement('a'),{href:url,download:`built_dataset.${fmt}`}).click();
      URL.revokeObjectURL(url);
      if (autoTriggered) setMsg(`Dataset ready. ${fmt.toUpperCase()} downloaded to your PC.`);
    } catch(e){ setMsg(e.message); }
  }

  const isDone = jobData?.status === 'done';

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Build dataset</h2>
        <p>LLM API keys are read from backend environment. Choose a build mode, configure options, then build from your selected connections.</p>
      </div>

      {selected.length === 0 && (
        <div style={{padding:'12px 16px',background:'rgba(251,191,36,0.07)',border:'1px solid rgba(251,191,36,0.25)',borderRadius:12,fontSize:13,color:'var(--yellow)',marginBottom:16}}>
          ⚠ No connections selected — go back and select at least one connection.
        </div>
      )}

      <div className="build-modes">
        {[['query','🔍 Query / Prompt'],['llm','🤖 LLM Build'],['manual','⚙️ Manual']].map(([m,l])=>(
          <button key={m} className={`mode-tab ${form.mode===m?'active':''}`} onClick={()=>set('mode',m)}>{l}</button>
        ))}
      </div>

      <div className="form-stack">
        {form.mode === 'query' && (
          <div className="form-group">
            <label>Target description or query</label>
            <textarea value={form.target_description} onChange={e=>set('target_description',e.target.value)}
              placeholder="e.g. Build a user-item interaction dataset for product recommendations based on purchase history" />
          </div>
        )}
        {form.mode === 'llm' && (
          <div className="form-group">
            <label>LLM build instructions</label>
            <textarea value={form.llm_prompt} onChange={e=>set('llm_prompt',e.target.value)}
              placeholder="Describe what kind of recommendation dataset you want the AI to build…" />
          </div>
        )}
        {form.mode === 'manual' && (
          <div className="form-stack">
            <div className="form-row">
              <div className="form-group"><label>Tables (comma-separated)</label><input value={form.manual_config.tables} onChange={e=>setManual('tables',e.target.value)} placeholder={availableTables.slice(0,2).join(',') || 'users,items,events'} /></div>
              <div className="form-group"><label>Target field</label><input value={form.manual_config.target_field} onChange={e=>setManual('target_field',e.target.value)} placeholder="user_id" /></div>
            </div>
            <div className="form-row">
              <div className="form-group"><label>Label field</label><input value={form.manual_config.label_field} onChange={e=>setManual('label_field',e.target.value)} placeholder="item_id" /></div>
              <div className="form-group"><label>Relationships</label><input value={form.manual_config.relationships} onChange={e=>setManual('relationships',e.target.value)} placeholder="users.id=events.user_id" /></div>
            </div>
            <div className="form-group"><label>Notes</label><textarea value={form.manual_config.notes} onChange={e=>setManual('notes',e.target.value)} placeholder="Any additional context…" style={{minHeight:60}} /></div>
          </div>
        )}

        <div className="form-row">
          <div className="form-group">
            <label>Recommendation type</label>
            <select value={form.rec_system_type} onChange={e=>set('rec_system_type',e.target.value)}>
              <option value="hybrid">Hybrid</option>
              <option value="collaborative">Collaborative Filtering</option>
              <option value="content_based">Content-Based</option>
              <option value="sequential">Sequential</option>
            </select>
          </div>
          <div className="form-group">
            <label>Output format</label>
            <select value={form.output_format} onChange={e=>set('output_format',e.target.value)}>
              <option value="csv">CSV</option>
              <option value="json">JSON</option>
            </select>
          </div>
        </div>
        <div className="form-group">
          <label>Max rows per table</label>
          <input type="number" value={form.max_rows_per_table} onChange={e=>set('max_rows_per_table',e.target.value)} />
        </div>

        <div className="actions">
          <button className="btn btn-primary" onClick={start} disabled={selected.length===0}>
            🚀 Start build
          </button>
          {job && isDone && (
            <>
              <button className="btn btn-secondary" onClick={()=>loadForTraining(false)}>→ Use for training</button>
              <button className="btn btn-ghost" onClick={download}>⬇ Download</button>
            </>
          )}
        </div>
      </div>

      {jobData && <JobStatus data={jobData} label="Build" />}

      {isDone && (
        <div className="log-box" style={{marginTop:12}}>
          {form.output_format === 'csv'
            ? 'CSV saved in the app, downloaded to your PC, and prepared for training.'
            : 'JSON downloaded to your PC.'}
          {jobData?.output_files?.[form.output_format] ? ` Backend copy: ${jobData.output_files[form.output_format]}` : ''}
        </div>
      )}

      {isDone && (
        <div className="advance-banner">
          <span className="advance-icon">✅</span>
          <div>
            <strong>Dataset ready!</strong> Automatically advancing to the Training step…
          </div>
        </div>
      )}
    </div>
  );
}

// ── PageTrain ─────────────────────────────────────────────────────────────────

function PageTrain({ file, setFile, onTrainFinished, onAdvance, setMsg }) {
  const [job, setJob] = useState(null);
  const [jobData, setJobData] = useState(null);
  const [form, setForm] = useState({
    user_col:'user_id',
    item_col:'item_id',
    rating_col:'rating',
    target_metric:'NDCG@K',
    top_k:10,
    test_size:0.2,
    n_trials:30,
    n_top_models:10,
    algorithm_mode:'auto',
    format:'auto',
  });
  const set = (k,v) => setForm(p=>({...p,[k]:v}));

  useEffect(()=>{
    if (!job) return;
    const t = setInterval(async()=>{
      try {
        const d = await jfetch(`${CORE_API}/jobs/${job}`);
        setJobData(d);
        if (isTerminal(d.status)) clearInterval(t);
      } catch(e){ setMsg(e.message); clearInterval(t); }
    }, 2000);
    return ()=>clearInterval(t);
  },[job]);

  useEffect(()=>{
    if (jobData?.status !== 'done') return;
    onTrainFinished(jobData);
  },[jobData?.status, jobData, onTrainFinished]);

  async function start() {
    if (!file) { setMsg('No dataset file — complete the Build step or upload a CSV.'); return; }
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('top_k', form.top_k);
      fd.append('n_trials', form.n_trials);
      fd.append('top_models', form.n_top_models);
      fd.append('algorithm_mode', form.algorithm_mode);
      fd.append('format', form.format);
      const res = await jfetch(`${CORE_API}/train/file`, { method:'POST', body:fd });
      setJob(res.job_id);
      setJobData({status:'pending',progress:0});
      setMsg('Training started.');
    } catch(e){ setMsg(e.message); }
  }

  async function pickFile(e) {
    const f = e.target.files?.[0];
    if (f) { setFile(f); setMsg(`File loaded: ${f.name}`); }
  }

  const isDone = jobData?.status === 'done';

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Train models</h2>
        <p>Optuna-powered hyperparameter search across multiple algorithms. The best model is promoted automatically.</p>
      </div>

      <div className="form-stack">
        <div>
          <div style={{padding:'16px',background:'var(--surface2)',border:`1px solid ${file?'rgba(52,211,153,0.3)':'var(--border)'}`,borderRadius:14,display:'flex',alignItems:'center',gap:14,transition:'border-color 0.2s'}}>
            <div style={{fontSize:28}}>📋</div>
            <div style={{flex:1,minWidth:0}}>
              <div style={{fontWeight:600,fontSize:14,marginBottom:2}}>{file ? file.name : 'No file selected'}</div>
              <div style={{fontSize:12,color:'var(--muted)'}}>
                {file ? `${(file.size/1024).toFixed(1)} KB` : 'Upload a CSV dataset or complete the Build step'}
              </div>
            </div>
            <label className="btn btn-secondary btn-sm" style={{cursor:'pointer'}}>
              Browse <input type="file" accept=".csv" onChange={pickFile} style={{display:'none'}} />
            </label>
          </div>
        </div>

        <div className="form-row3">
          <div className="form-group"><label>User column</label><input value={form.user_col} onChange={e=>set('user_col',e.target.value)} /></div>
          <div className="form-group"><label>Item column</label><input value={form.item_col} onChange={e=>set('item_col',e.target.value)} /></div>
          <div className="form-group"><label>Rating column</label><input value={form.rating_col} onChange={e=>set('rating_col',e.target.value)} /></div>
        </div>
        <div className="form-row">
          <div className="form-group">
            <label>Target metric</label>
            <select value={form.target_metric} onChange={e=>set('target_metric',e.target.value)}>
              {['NDCG@K','Precision@K','Recall@K','MAP@K','RMSE'].map(m=><option key={m} value={m}>{m}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label>Top K</label>
            <select value={form.top_k} onChange={e=>set('top_k',e.target.value)}>
              <option value="10">Top 10</option>
              <option value="5">Top 5</option>
            </select>
          </div>
        </div>
        <div className="form-row3">
          <div className="form-group"><label>Test size</label><input type="number" value={form.test_size} onChange={e=>set('test_size',e.target.value)} step="0.05" min="0.05" max="0.5" /></div>
          <div className="form-group"><label>Optuna trials</label><input type="number" value={form.n_trials} onChange={e=>set('n_trials',e.target.value)} min={5} max={200} /></div>
          <div className="form-group">
            <label>Top models</label>
            <select value={form.n_top_models} onChange={e=>set('n_top_models',e.target.value)}>
              <option value="10">Top 10</option>
              <option value="5">Top 5</option>
            </select>
          </div>
        </div>
        <div className="form-group">
          <label>Algorithm mode</label>
          <select value={form.algorithm_mode} onChange={e=>set('algorithm_mode',e.target.value)}>
            <option value="auto">Auto detect</option>
            <option value="explicit">Explicit only</option>
            <option value="implicit">Implicit only</option>
          </select>
          <div style={{fontSize:12,color:'var(--muted)',marginTop:6,lineHeight:1.5}}>
            Auto detects the dataset type. Explicit runs only explicit-feedback algorithms. Implicit runs only implicit-feedback algorithms.
          </div>
        </div>

        <div className="actions">
          <button className="btn btn-primary" onClick={start} disabled={!file}>
            🧠 Start training
          </button>
        </div>
      </div>

      {jobData && <JobStatus data={jobData} label="Training" />}

      {isDone && (
        <div className="advance-banner">
          <div>
            <strong>Training complete!</strong>
            <span style={{display:'block',marginTop:4}}>Use Next to continue to Recommendations.</span>
          </div>
          <button className="btn btn-primary btn-sm" onClick={onAdvance}>Next</button>
        </div>
      )}

    </div>
  );
}

// ── PageRecommend ─────────────────────────────────────────────────────────────

function PageRecommend({ setMsg, trainingResult }) {
  const training = trainingResult?.result || null;
  const [loginForm, setLoginForm] = useState({ username:'admin', password:'admin123' });
  const [token, setToken] = useState(()=> localStorage.getItem(TOKEN_KEY) || '');
  const [options, setOptions] = useState(training?.recommendation_options || null);
  const [rankingLimit, setRankingLimit] = useState(1);
  const [form, setForm] = useState({
    user_id:'',
    top_n:10,
    model_id: training?.best_model_id || '',
    strategy:'best_promoted_model',
  });
  const set = (k,v) => setForm(p=>({...p,[k]:v}));
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [authBusy, setAuthBusy] = useState(false);

  const models = useMemo(() => uniqueModels(
    options?.supported_models
    || options?.single_model_options
    || training?.recommendation_options?.supported_models
    || training?.recommendation_options?.single_model_options
    || training?.model_selection_policy?.selected_models
    || []
  ), [options, training]);
  const recommendedModels = options?.recommended_models
    || training?.recommendation_options?.recommended_models
    || training?.model_selection_policy?.serving_selected_models
    || training?.model_selection_policy?.selected_models
    || [];
  const rankedModels = options?.ranked_models
    || training?.recommendation_options?.ranked_models
    || training?.model_selection_policy?.selected_models
    || [];
  const supportedModelCount = options?.supported_model_count
    ?? training?.recommendation_options?.supported_model_count
    ?? models.length;
  const bestPromoted = options?.best_promoted_model || null;
  const bestModelExplanation = options?.best_model_explanation
    || training?.recommendation_options?.best_model_explanation
    || training?.model_selection_policy?.best_model_explanation
    || training?.model_selection_policy?.reason
    || '';
  const recommendations = result?.recommendations || result?.items || [];
  const warnings = result?.warnings || [];
  const contributionBreakdown = result?.contribution_breakdown || [];

  useEffect(()=>{
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  },[token]);

  useEffect(()=>{
    if (training?.recommendation_options) setOptions(training.recommendation_options);
    setRankingLimit(1);
    if (training?.best_model_id) {
      setForm(prev => prev.model_id ? prev : { ...prev, model_id: training.best_model_id });
    }
  },[training]);

  useEffect(()=>{
    if (!token) return;
    loadRecommendationContext(token);
  },[token, rankingLimit]);

  async function loadRecommendationContext(activeToken = token) {
    try {
      const headers = { Authorization:`Bearer ${activeToken}` };
      const opt = await jfetch(`${CORE_API}/recommend/options?top_n_models=${rankingLimit}`, { headers });
      setOptions(opt);
      if (!form.model_id && opt?.best_promoted_model?.model_id) {
        setForm(prev => prev.model_id ? prev : { ...prev, model_id: opt.best_promoted_model.model_id });
      }
    } catch (e) {
      const message = String(e.message || '');
      if (message.toLowerCase().includes('token') || message.toLowerCase().includes('bearer')) {
        setToken('');
      }
      setMsg(message);
    }
  }

  async function login() {
    if (!loginForm.username.trim() || !loginForm.password.trim()) {
      setMsg('Enter username and password to unlock recommendations.');
      return;
    }
    setAuthBusy(true);
    try {
      const auth = await jfetch(`${CORE_API}/auth/login`, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          username: loginForm.username.trim(),
          password: loginForm.password,
        }),
      });
      setToken(auth.access_token || '');
      setMsg('Recommendation access is ready.');
    } catch (e) {
      setMsg(e.message);
    } finally {
      setAuthBusy(false);
    }
  }

  async function query() {
    if (!form.user_id.trim()) { setMsg('Please enter a user ID.'); return; }
    if (!token) { setMsg('Login first to call the recommendation API.'); return; }
    setLoading(true);
    try {
      const strategy = form.strategy === 'single_model' && !form.model_id ? 'best_promoted_model' : form.strategy;
      const payload = {
        user_id: form.user_id.trim(),
        top_n: Number(form.top_n) || 10,
        strategy,
      };

      if (strategy === 'single_model') {
        const chosen = models.find(model => model.model_id === form.model_id);
        if (!chosen?.model_id) throw new Error('Choose a trained model before using single model mode.');
        payload.model_id = chosen.model_id;
        if (chosen.algorithm) payload.algorithm = chosen.algorithm;
      }

      if (strategy === 'ensemble_weighted') {
        const ensemble = recommendedModels
          .filter(model => model.model_id || model.algorithm)
          .map((model, index) => ({
            model_id: model.model_id || null,
            algorithm: model.algorithm,
            weight: Number(model.selection_score_pct || model.normalized_weight || 0) || (recommendedModels.length - index),
          }));
        if (ensemble.length < 2) {
          throw new Error('This training run does not have enough eligible models for ensemble recommendations.');
        }
        payload.models = ensemble;
      }

      const d = await jfetch(`${CORE_API}/recommend`, {
        method:'POST',
        headers:{
          'Content-Type':'application/json',
          Authorization:`Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      });
      setResult(d);
    } catch(e){ setMsg(e.message); }
    finally { setLoading(false); }
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Get recommendations</h2>
        <p>Use the trained model set to generate personalised recommendations for any user.</p>
      </div>

      {training && (
        <div style={{marginBottom:20,padding:'14px 16px',background:'var(--surface2)',borderRadius:14,border:'1px solid var(--border)'}}>
          <div style={{fontSize:11,fontWeight:700,color:'var(--muted)',textTransform:'uppercase',letterSpacing:'0.06em',marginBottom:8}}>Training summary</div>
          <div style={{display:'flex',gap:8,flexWrap:'wrap'}}>
            {training.best_algorithm && <span className="badge badge-green">Best: {training.best_algorithm}</span>}
            {training.metric && <span className="badge badge-blue">{training.target_metric} {num(training.metric)}</span>}
            {supportedModelCount > 0 && <span className="badge">{supportedModelCount} API-supported models</span>}
            {recommendedModels.length > 0 && <span className="badge badge-blue">Default serving set: {recommendedModels.length}</span>}
          </div>
          {bestModelExplanation && (
            <div style={{fontSize:13,lineHeight:1.6,color:'var(--muted)',marginTop:10}}>
              {bestModelExplanation}
            </div>
          )}
        </div>
      )}

      {supportedModelCount > 0 && (
        <div style={{marginBottom:20,padding:'14px 16px',background:'var(--surface2)',borderRadius:14,border:'1px solid var(--border)'}}>
          <div style={{display:'flex',justifyContent:'space-between',gap:12,flexWrap:'wrap',alignItems:'end'}}>
            <div>
              <div style={{fontSize:11,fontWeight:700,color:'var(--muted)',textTransform:'uppercase',letterSpacing:'0.06em',marginBottom:8}}>Model ranking</div>
              <div style={{fontSize:13,color:'var(--muted)',lineHeight:1.6}}>
                Showing only models that the recommendation API can actually serve.
              </div>
            </div>
            <div className="form-group" style={{minWidth:180,marginBottom:0}}>
              <label>Explain top N models</label>
              <input
                type="number"
                min={1}
                max={Math.max(1, supportedModelCount)}
                value={rankingLimit}
                onChange={e=>setRankingLimit(Math.max(1, Math.min(Math.max(1, supportedModelCount), Number(e.target.value) || 1)))}
              />
            </div>
          </div>
          <div style={{display:'flex',gap:8,flexWrap:'wrap',marginTop:12}}>
            <span className="badge badge-blue">Visible ranks: {rankedModels.length}</span>
            {bestPromoted?.algorithm && <span className="badge">Promoted default: {bestPromoted.algorithm}</span>}
          </div>
          <div className="cards-stack" style={{marginTop:14}}>
            {rankedModels.map(model=>(
              <div key={model.model_id||model.algorithm} className="rec-card">
                <div className="rec-card-header">
                  <div className="rec-card-rank">Top {model.rank}: {model.algorithm}</div>
                  <span className="badge badge-green">Score {pct(model.selection_score_pct)}</span>
                </div>
                <div style={{fontSize:13,lineHeight:1.6,color:'var(--muted)'}}>
                  {model.reason || model.summary || 'No ranking explanation was provided.'}
                </div>
                {model.comparison_to_next && (
                  <div style={{fontSize:12,lineHeight:1.6,color:'var(--text)',marginTop:8}}>
                    Why #{model.rank} beats #{model.rank + 1}: {model.comparison_to_next}
                  </div>
                )}
                {model.metric_name && model.metric_value != null && (
                  <div style={{fontSize:11,color:'var(--muted)',marginTop:8,fontFamily:'DM Mono,monospace'}}>
                    {model.metric_name} {num(model.metric_value)} {model.composite_score != null ? `· Composite ${num(model.composite_score)}` : ''}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="form-stack">
        {!token ? (
          <div className="panel" style={{padding:'16px'}}>
            <div style={{fontSize:12,fontWeight:700,color:'var(--muted)',marginBottom:10,textTransform:'uppercase',letterSpacing:'0.06em'}}>Recommendation login</div>
            <div className="form-row">
              <div className="form-group">
                <label>Username</label>
                <input value={loginForm.username} onChange={e=>setLoginForm(prev=>({...prev, username:e.target.value}))} />
              </div>
              <div className="form-group">
                <label>Password</label>
                <input type="password" value={loginForm.password} onChange={e=>setLoginForm(prev=>({...prev, password:e.target.value}))} onKeyDown={e=>e.key==='Enter'&&login()} />
              </div>
            </div>
            <div className="actions">
              <button className="btn btn-secondary" onClick={login} disabled={authBusy}>
                {authBusy ? 'Signing in...' : 'Unlock recommendations'}
              </button>
            </div>
          </div>
        ) : (
          <div style={{padding:'12px 14px',background:'rgba(52,211,153,0.06)',border:'1px solid rgba(52,211,153,0.25)',borderRadius:12,fontSize:13}}>
            Recommendation API connected.
            {bestPromoted?.algorithm ? ` Current promoted model: ${bestPromoted.algorithm}.` : ''}
          </div>
        )}

        <div className="form-row">
          <div className="form-group">
            <label>User ID</label>
            <input value={form.user_id} onChange={e=>set('user_id',e.target.value)} placeholder="e.g. user_42" onKeyDown={e=>e.key==='Enter'&&query()} />
          </div>
          <div className="form-group">
            <label>Number of results</label>
            <input type="number" value={form.top_n} onChange={e=>set('top_n',e.target.value)} min={1} max={100} />
          </div>
        </div>
        <div className="form-row">
          <div className="form-group">
            <label>Recommendation mode</label>
            <select value={form.strategy} onChange={e=>set('strategy',e.target.value)}>
              <option value="best_promoted_model">Use promoted best model</option>
              <option value="single_model">Pick one trained model</option>
              <option value="ensemble_weighted">Blend shortlisted models</option>
            </select>
          </div>
          <div className="form-group">
            <label>Model override</label>
            <select value={form.model_id} onChange={e=>set('model_id',e.target.value)}>
              <option value="">Use promoted best model</option>
              {models.map(m=>(
                <option key={m.model_id||m.algorithm} value={m.model_id||''}>
                  {m.algorithm}{m.promoted ? ' (promoted)' : ''}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div style={{fontSize:13,color:'var(--muted)',lineHeight:1.6}}>
          {form.strategy === 'best_promoted_model' && 'Best for most users: it uses the promoted model that training marked as the safest default choice.'}
          {form.strategy === 'single_model' && 'Best for comparison: it runs one exact trained model so you can inspect that model on its own.'}
          {form.strategy === 'ensemble_weighted' && `Best for broader coverage: it blends the default serving set${recommendedModels.length ? ` (${recommendedModels.length} available)` : ''}.`}
        </div>
        <div className="actions">
          <button className="btn btn-primary" onClick={query} disabled={loading || !token}>
            {loading ? '⏳ Querying…' : '✨ Get recommendations'}
          </button>
        </div>
      </div>

      {loading && <LoadingSpinner label="Fetching recommendations…" />}

      {result && !loading && (
        <div style={{marginTop:24,display:'flex',flexDirection:'column',gap:12,animation:'fadeUp 0.35s ease'}}>
          <div style={{padding:'14px 16px',background:'var(--surface2)',borderRadius:14,border:'1px solid var(--border)'}}>
            <strong>Summary:</strong> {recommendations.length} items using {result.strategy === 'ensemble_weighted' ? 'weighted ensemble' : (result.algorithm||'selected model')}.
          </div>

          {warnings.length > 0 && warnings.map((w,i)=>(
            <div key={i} style={{padding:'12px 14px',background:'rgba(251,191,36,0.06)',border:'1px solid rgba(251,191,36,0.2)',borderRadius:12,fontSize:13,color:'var(--yellow)'}}>
              ⚠ {recText(w)}
            </div>
          ))}

          {contributionBreakdown.length > 0 && (
            <div className="mini-grid">
              {contributionBreakdown.map(row=>(
                <div key={row.model_id||row.algorithm} style={{background:'var(--surface2)',border:'1px solid var(--border)',borderRadius:14,padding:'14px 16px'}}>
                  <div style={{fontFamily:'Syne,sans-serif',fontWeight:700,fontSize:14,marginBottom:6}}>{row.algorithm}</div>
                  <div style={{fontSize:12,color:'var(--muted)',lineHeight:1.5}}>
                    Influenced {row.item_count||0} items · {pct(row.share_pct)} of ranking
                  </div>
                </div>
              ))}
            </div>
          )}

          {recommendations.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon">🎯</div>
              <div>No recommendations returned for this user.</div>
            </div>
          )}

          <div className="cards-stack">
            {recommendations.map(item=>(
              <div key={`${item.item_id}-${item.rank}`} className="rec-card">
                <div className="rec-card-header">
                  <div className="rec-card-rank">#{item.rank} Item {item.item_id}</div>
                  <span className="badge badge-green">Score {num(item.final_score??item.score, 6)}</span>
                </div>
                {item.explanation && <div style={{fontSize:13,lineHeight:1.5,color:'var(--muted)'}}>{item.explanation}</div>}
                {item.contributions?.length > 0 && (
                  <div style={{fontSize:11,color:'var(--muted)',marginTop:8,fontFamily:'DM Mono,monospace'}}>
                    {item.contributions.slice(0,3).map(p=>`${p.algorithm} ${pct(p.share_pct)}`).join(' · ')}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {!result && !loading && (
        <div className="empty-state" style={{marginTop:24}}>
          <div className="empty-icon">🎯</div>
          <div>Enter a user ID above and hit Get recommendations</div>
        </div>
      )}
    </div>
  );
}

// ── App Shell ─────────────────────────────────────────────────────────────────

export default function App() {
  const [mode, setMode] = useState(null);
  const [page, setPage] = useState('mode');
  const [connections, setConnections] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [schemas, setSchemas] = useState({});
  const [builtFile, setBuiltFile] = useState(null);
  const [trainingResult, setTrainingResult] = useState(null);
  const [msg, setMsg] = useState('');
  const [donePages, setDonePages] = useState(new Set());

  useEffect(()=>{ loadConnections(); }, []);

  async function loadConnections(autoSelectId) {
    try {
      const raw = await jfetch(`${SMART_API}/connections`);
      const list = raw.map(norm);
      setConnections(list);
      if (autoSelectId) setSelectedIds(p=>p.includes(autoSelectId)?p:[...p,autoSelectId]);
    } catch(e){ setMsg(e.message); }
  }

  async function removeConnection(id) {
    try {
      await jfetch(`${SMART_API}/connections/${id}`,{method:'DELETE'});
      setSelectedIds(p=>p.filter(x=>x!==id));
      setSchemas(p=>{ const n={...p}; delete n[id]; return n; });
      await loadConnections();
    } catch(e){ setMsg(e.message); }
  }

  async function loadSchema(id) {
    if (schemas[id]) return;
    try { const d = await jfetch(`${SMART_API}/schema/${id}`); setSchemas(p=>({...p,[id]:d})); }
    catch(e){ setMsg(e.message); }
  }

  function markDone(p) { setDonePages(prev => new Set([...prev, p])); }

  function selectMode(m) {
    setMode(m);
    markDone('mode');
    setPage(m==='csv' ? 'train' : 'connections');
  }

  const PAGES_DB  = ['mode','connections','build','train','recommend'];
  const PAGES_CSV = ['mode','train','recommend'];
  const pages = mode==='csv' ? PAGES_CSV : PAGES_DB;
  const labels = { mode:'Source', connections:'Connections', build:'Build', train:'Train', recommend:'Recommend' };
  const stepIcons = { mode:'◎', connections:'🔗', build:'🛠', train:'🧠', recommend:'✨' };

  return (
    <>
      <style>{css}</style>
      <div className="app-wrap">

        {/* Header */}
        <div className="header">
          <div className="logo-area">
            <div className="logo-tag"><span className="dot" />{mode ? (mode==='csv'?'CSV Mode':'Database Mode') : 'Proactive AI'}</div>
    
            <p>Train recommendation models from a CSV or live database, then query the promoted model — all in one seamless workflow.</p>
          </div>
          {mode && (
            <button className="btn btn-ghost btn-sm" onClick={()=>{ setMode(null); setPage('mode'); setDonePages(new Set()); }}>
              ↩ Change source
            </button>
          )}
        </div>

        {/* Step indicator */}
        <div className="stepper">
          {pages.map((p, i) => (
            <div key={p} className="step-item">
              {i > 0 && <div className="step-connector" />}
              <button
                className={`step-btn ${page===p?'active':''} ${donePages.has(p)&&page!==p?'done':''}`}
                onClick={()=>setPage(p)}
              >
                <span className="step-num">{donePages.has(p)&&page!==p ? '✓' : i+1}</span>
                {stepIcons[p]} {labels[p]}
              </button>
            </div>
          ))}
        </div>

        <Msg msg={msg} clear={()=>setMsg('')} />

        {page==='mode' && <PageMode onSelect={selectMode} />}

        {page==='connections' && (
          <PageConnections
            connections={connections.map(norm)}
            selected={selectedIds}
            schemas={schemas}
            onAdded={id=>{ loadConnections(id); markDone('connections'); }}
            onRemove={removeConnection}
            onToggle={(id,checked)=>setSelectedIds(p=>checked?[...p,id]:p.filter(x=>x!==id))}
            onSchema={loadSchema}
            onAdvance={()=>{ markDone('connections'); setPage('build'); setMsg('Connections selected. Continue with dataset build.'); }}
            setMsg={setMsg}
          />
        )}

        {page==='build' && (
          <PageBuild
            selected={selectedIds}
            schemas={schemas}
            onBuilt={f=>{ setBuiltFile(f); markDone('build'); setPage('train'); setMsg('Dataset ready — proceeding to Train. ✓'); }}
            setMsg={setMsg}
          />
        )}

        {page==='train' && (
          <PageTrain
            file={builtFile}
            setFile={setBuiltFile}
            onTrainFinished={r=>{ setTrainingResult(r); markDone('train'); setMsg('Training complete. Review the result, then use Next to continue.'); }}
            onAdvance={()=>{ setPage('recommend'); setMsg('Training result loaded. Continue with recommendations.'); }}
            setMsg={setMsg}
          />
        )}

        {page==='recommend' && <PageRecommend setMsg={setMsg} trainingResult={trainingResult} />}
      </div>
    </>
  );
}
