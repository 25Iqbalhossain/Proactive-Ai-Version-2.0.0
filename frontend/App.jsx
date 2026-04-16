import React, { useEffect, useMemo, useState } from 'react';

const SMART = '/smart-db-csv/api';

const DB_OPTS = [
  ['mysql','MySQL / MariaDB'],['postgres','PostgreSQL'],
  ['mssql','SQL Server'],['mongodb','MongoDB'],['sqlite','SQLite'],
];

const PORT_MAP = { postgres:'5432', mysql:'3306', mssql:'1433', mongodb:'27017', sqlite:'' };

const css = `
  :root{--bg:#0b1020;--panel:#121a2d;--soft:#1b2742;--line:#2a3960;--text:#eef3ff;--muted:#9bb0d1;--accent:#6ea8fe;--ok:#42d392;--warn:#f7b955;--err:#ff7b7b;}
  *{box-sizing:border-box;}body{margin:0;background:linear-gradient(180deg,#0b1020,#101a30);color:var(--text);font-family:Inter,Segoe UI,Arial,sans-serif;}
  .wrap{max-width:860px;margin:0 auto;padding:24px;}
  .hero{margin-bottom:24px;}.hero h1{margin:0 0 6px;font-size:26px;}.hero p{margin:0;color:var(--muted);line-height:1.5;font-size:14px;}
  .steps{display:flex;gap:8px;margin-bottom:24px;flex-wrap:wrap;}
  .step-btn{padding:8px 16px;border-radius:999px;border:1px solid var(--line);background:transparent;color:var(--muted);font-size:13px;cursor:pointer;font-weight:600;}
  .step-btn.active{background:var(--accent);color:#08111f;border-color:var(--accent);}
  .panel{background:rgba(18,26,45,.94);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:0 12px 40px rgba(0,0,0,.18);margin-bottom:16px;}
  .panel h2{margin:0 0 4px;font-size:18px;}.sub{color:var(--muted);font-size:13px;margin-bottom:16px;line-height:1.45;}
  .stack{display:grid;gap:14px;}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px;}.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;}
  label{display:block;font-size:12px;color:var(--muted);margin-bottom:5px;}
  input,select,textarea{width:100%;background:var(--soft);border:1px solid var(--line);color:var(--text);border-radius:12px;padding:10px 12px;font-size:14px;}
  textarea{min-height:80px;resize:vertical;}
  button{border:none;background:var(--accent);color:#08111f;font-weight:700;padding:10px 18px;border-radius:12px;cursor:pointer;font-size:14px;}
  button.sec{background:transparent;color:var(--text);border:1px solid var(--line);}
  button.ghost{background:#1a2440;color:var(--text);}
  button:disabled{opacity:.5;cursor:not-allowed;}
  .actions{display:flex;gap:10px;flex-wrap:wrap;align-items:center;}
  .badge{display:inline-flex;padding:5px 10px;border-radius:999px;border:1px solid var(--line);background:#131d34;color:var(--muted);font-size:12px;font-weight:700;}
  .badge.ok{color:var(--ok);border-color:rgba(66,211,146,.3);}
  .badge.warn{color:var(--warn);border-color:rgba(247,185,85,.3);}
  .badge.err{color:var(--err);border-color:rgba(255,123,123,.3);}
  .cards{display:grid;gap:10px;}
  .conn{border:1px solid var(--line);border-radius:14px;padding:12px;background:#11192c;display:flex;justify-content:space-between;align-items:center;gap:12px;}
  .muted{color:var(--muted);}.tiny{font-size:12px;}.code{font-family:ui-monospace,monospace;font-size:12px;}
  .log{background:#0d1426;border:1px solid var(--line);border-radius:14px;padding:12px;overflow:auto;max-height:320px;white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:12px;}
  .choice{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:8px;}
  .choice-card{border:2px solid var(--line);border-radius:16px;padding:20px;cursor:pointer;text-align:center;transition:border-color .2s,background .2s;}
  .choice-card:hover,.choice-card.sel{border-color:var(--accent);background:rgba(110,168,254,.07);}
  .choice-card h3{margin:8px 0 4px;font-size:16px;}.choice-card p{margin:0;color:var(--muted);font-size:13px;}
  .msg{border:1px solid var(--line);border-radius:12px;padding:10px 14px;background:#131d34;margin-bottom:12px;font-size:13px;display:flex;justify-content:space-between;align-items:center;gap:8px;}
  @media(max-width:700px){.row,.row3,.choice{grid-template-columns:1fr;}}
`;

async function jfetch(url, opts = {}) {
  const res = await fetch(url, opts);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!res.ok) throw new Error((data && (data.detail || data.message || data.error)) || text || `HTTP ${res.status}`);
  return data;
}

function Msg({ msg, clear }) {
  if (!msg) return null;
  return (
    <div className="msg">
      <span>{msg}</span>
      <button className="sec" style={{padding:'3px 10px',fontSize:11}} onClick={clear}>✕</button>
    </div>
  );
}

function JobStatus({ data, label }) {
  if (!data) return null;
  const cls = data.status === 'done' ? 'ok' : data.status === 'failed' ? 'err' : 'warn';
  return (
    <div>
      <div style={{display:'flex',gap:8,flexWrap:'wrap',marginBottom:8}}>
        <span className={`badge ${cls}`}>{label} — {data.status}</span>
        {data.progress != null && <span className="badge">Progress {data.progress}%</span>}
        {data.row_count && <span className="badge">Rows {data.row_count}</span>}
        {data.result?.best_algorithm && <span className="badge ok">Best: {data.result.best_algorithm}</span>}
      </div>
      <div className="log">{JSON.stringify(data.plan || data.steps || data, null, 2)}</div>
    </div>
  );
}

// normalise connection records — backend may nest attrs under .cred
function norm(c) {
  return {
    ...c,
    name: c.name || (c.cred && c.cred.name) || c.id,
    db_type: c.db_type || (c.cred && c.cred.db_type) || '?',
    database: c.database || (c.cred && c.cred.database) || '',
  };
}

// ── pages ────────────────────────────────────────────────────────────────────

function PageMode({ onSelect }) {
  const [hover, setHover] = useState('');
  const cards = [
    ['csv','📄','Upload CSV','Directly upload an existing CSV to train recommendation models.'],
    ['db','🗄️','Database','Connect to a DB, build a dataset with the smart planner, then train.'],
  ];
  return (
    <div className="panel">
      <h2>Choose data source</h2>
      <p className="sub">How would you like to supply training data?</p>
      <div className="choice">
        {cards.map(([k,icon,title,desc]) => (
          <div key={k} className={`choice-card ${hover===k?'sel':''}`}
            onMouseEnter={()=>setHover(k)} onMouseLeave={()=>setHover('')}
            onClick={()=>onSelect(k)}>
            <div style={{fontSize:34}}>{icon}</div>
            <h3>{title}</h3><p>{desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

async function save() {
  try {
    const payload = {
      ...form,
      port: form.port ? Number(form.port) : null,
    };

    await jfetch(`${SMART}/connections/test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const res = await jfetch(`${SMART}/connections`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    setMsg(`Connected: ${res.name}`);
    setForm(blank);
    onAdded(res.id);
  } catch (e) {
    setMsg(e.message);
  }
}
function PageConnections({ connections, selected, schemas, onAdded, onRemove, onToggle, onSchema, setMsg }) {
  const blank = { db_type:'postgres', name:'', host:'', port:'5432', database:'', username:'', password:'', filepath:'', uri:'' };
  const [form, setForm] = useState(blank);
  const set = (k,v) => setForm(p=>({...p,[k]:v}));

  function changeType(t) { setForm(p=>({...p, db_type:t, port: PORT_MAP[t]||''})); }

  async function save() {
    try {
      const payload = { ...form, port: form.port ? Number(form.port) : null };
      const res = await jfetch(`${SMART}/connections`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
      setMsg(`Connected: ${res.name}`);
      setForm(blank);
      onAdded(res.id); // auto-select the new connection
    } catch(e) { setMsg(e.message); }
  }

  return (
    <div className="stack">
      <div className="panel">
        <h2>Add connection</h2>
        <p className="sub">Credentials are stored on the backend. New connections are auto-selected.</p>
        <div className="stack">
          <div><label>Database type</label>
            <select value={form.db_type} onChange={e=>changeType(e.target.value)}>
              {DB_OPTS.map(([v,l])=><option key={v} value={v}>{l}</option>)}
            </select>
          </div>
          <div className="row">
            <div><label>Connection name</label><input value={form.name} onChange={e=>set('name',e.target.value)} placeholder="sales-db" /></div>
            <div><label>Database</label><input value={form.database} onChange={e=>set('database',e.target.value)} placeholder="analytics" /></div>
          </div>
          {form.db_type !== 'sqlite' && (
            <div className="row3">
              <div><label>Host</label><input value={form.host} onChange={e=>set('host',e.target.value)} placeholder="localhost" /></div>
              <div><label>Port</label><input value={form.port} onChange={e=>set('port',e.target.value)} /></div>
              <div><label>User</label><input value={form.username} onChange={e=>set('username',e.target.value)} /></div>
            </div>
          )}
          {form.db_type === 'sqlite'
            ? <div><label>SQLite file path</label><input value={form.filepath} onChange={e=>set('filepath',e.target.value)} placeholder="/data/db.sqlite" /></div>
            : <div><label>Password</label><input type="password" value={form.password} onChange={e=>set('password',e.target.value)} /></div>}
          {form.db_type === 'mongodb' && <div><label>MongoDB URI (optional)</label><input value={form.uri} onChange={e=>set('uri',e.target.value)} placeholder="mongodb://user:pass@host:27017/db" /></div>}
          <div className="actions"><button onClick={save}>Save connection</button></div>
        </div>
      </div>

      <div className="panel">
        <h2>Saved connections</h2>
        <p className="sub">Check to include in the build step.</p>
        <div className="cards">
          {connections.length === 0 && <div className="muted tiny">No connections yet.</div>}
          {connections.map(c=>(
            <div key={c.id} className="conn">
              <div style={{display:'flex',gap:10,alignItems:'center'}}>
                <input type="checkbox" checked={selected.includes(c.id)}
                  onChange={e=>{ onToggle(c.id, e.target.checked); if(e.target.checked) onSchema(c.id); }} />
                <div>
                  <strong>{c.name}</strong> <span className="badge ok" style={{marginLeft:4}}>{c.db_type}</span>
                  <div className="tiny muted">{c.database||'—'}</div>
                </div>
              </div>
              <div className="actions">
                <button className="ghost" style={{padding:'6px 12px'}} onClick={()=>onSchema(c.id)}>Schema</button>
                <button className="sec" style={{padding:'6px 12px'}} onClick={()=>onRemove(c.id)}>Remove</button>
              </div>
            </div>
          ))}
        </div>
        {Object.entries(schemas).map(([id,s])=>(
          <div key={id} style={{marginTop:12}}>
            <div className="tiny muted" style={{marginBottom:4}}>{connections.find(c=>c.id===id)?.name||id} — schema</div>
            <div className="log">{JSON.stringify((s.tables||[]).slice(0,8).map(t=>({table:t.table_name,cols:(t.columns||[]).map(c=>`${c.name}:${c.data_type}`)})),null,2)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function PageBuild({ selected, onBuilt, setMsg }) {
  const [form, setForm] = useState({ rec_system_type:'hybrid', output_format:'csv', max_rows_per_table:50000, target_description:'' });
  const [job, setJob] = useState(null);
  const [jobData, setJobData] = useState(null);
  const set = (k,v) => setForm(p=>({...p,[k]:v}));

  useEffect(()=>{
    if (!job) return;
    const t = setInterval(async()=>{
      try {
        const d = await jfetch(`${SMART}/jobs/${job}`);
        setJobData(d);
        if (d.status==='done'||d.status==='failed') clearInterval(t);
      } catch(e){ setMsg(e.message); clearInterval(t); }
    }, 2000);
    return ()=>clearInterval(t);
  },[job]);

  async function start() {
    try {
      const payload = { ...form, connection_ids:selected, max_rows_per_table:Number(form.max_rows_per_table) };
      const res = await jfetch(`${SMART}/build`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
      setJob(res.job_id); setJobData({status:'pending',progress:0});
      setMsg('Dataset build started.');
    } catch(e){ setMsg(e.message); }
  }

  async function loadForTraining() {
    try {
      const res = await fetch(`${SMART}/jobs/${job}/download?output_format=csv`);
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      onBuilt(new File([blob],'built_dataset.csv',{type:'text/csv'}));
    } catch(e){ setMsg(e.message); }
  }

  async function download() {
    try {
      const fmt = form.output_format||'csv';
      const res = await fetch(`${SMART}/jobs/${job}/download?output_format=${fmt}`);
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      Object.assign(document.createElement('a'),{href:url,download:`built_dataset.${fmt}`}).click();
      URL.revokeObjectURL(url);
    } catch(e){ setMsg(e.message); }
  }

  return (
    <div className="panel">
      <h2>Build dataset</h2>
      <p className="sub">LLM API keys are read from backend environment variables — nothing to enter here. Description is optional; the planner infers from schema.</p>
      <div className="stack">
        <div className="row">
          <div><label>Recommendation type</label>
            <select value={form.rec_system_type} onChange={e=>set('rec_system_type',e.target.value)}>
              <option value="hybrid">Hybrid</option><option value="collaborative">Collaborative</option>
              <option value="content_based">Content based</option><option value="sequential">Sequential</option>
            </select>
          </div>
          <div><label>Output format</label>
            <select value={form.output_format} onChange={e=>set('output_format',e.target.value)}>
              <option value="csv">CSV</option><option value="json">JSON</option>
            </select>
          </div>
        </div>
        <div><label>Target description / query (optional)</label>
          <textarea value={form.target_description} onChange={e=>set('target_description',e.target.value)}
            placeholder="e.g. Build an e-commerce recommendation dataset from users, orders, and products. Leave blank to auto-infer from schema." />
        </div>
        <div><label>Max rows per table</label>
          <input type="number" value={form.max_rows_per_table} onChange={e=>set('max_rows_per_table',e.target.value)} />
        </div>
        <div className="actions">
          <button disabled={selected.length===0} onClick={start}>Build dataset</button>
          {jobData?.status==='done' && <button className="sec" onClick={download}>Download</button>}
          {jobData?.status==='done' && <button className="ghost" onClick={loadForTraining}>Load for training →</button>}
        </div>
        <JobStatus data={jobData} label="Build" />
      </div>
    </div>
  );
}

function PageTrain({ file, setFile, onTrained, setMsg }) {
  const [form, setForm] = useState({ top_k:10, n_trials:-1, top_models:10, algorithm_mode:'auto' });
  const [job, setJob] = useState(null);
  const [jobData, setJobData] = useState(null);
  const set = (k,v) => setForm(p=>({...p,[k]:v}));

  useEffect(()=>{
    if (!job) return;
    const t = setInterval(async()=>{
      try {
        const d = await jfetch(`/jobs/${job}`);
        setJobData(d);
        if (d.status==='done'||d.status==='failed'){ clearInterval(t); if(d.status==='done') onTrained(); }
      } catch(e){ setMsg(e.message); clearInterval(t); }
    }, 2500);
    return ()=>clearInterval(t);
  },[job]);

  async function train() {
    if (!file) return;
    try {
      const fd = new FormData();
      fd.append('file', file, file.name||'dataset.csv');
      fd.append('top_k', String(form.top_k));
      fd.append('n_trials', String(form.n_trials));
      fd.append('top_models', String(form.top_models));
      fd.append('algorithm_mode', form.algorithm_mode);
      fd.append('format', 'csv');
      const res = await fetch('/train/file',{method:'POST',body:fd});
      const text = await res.text();
      const data = text ? JSON.parse(text) : null;
      if (!res.ok) throw new Error(data?.detail||text||`HTTP ${res.status}`);
      setJob(data.job_id); setJobData({status:'pending'}); setMsg('Training started.');
    } catch(e){ setMsg(e.message); }
  }

  return (
    <div className="panel">
      <h2>Train recommendation models</h2>
      <p className="sub">Upload a CSV, or use a dataset built from a database in the previous step.</p>
      <div className="stack">
        <div>
          <label>Dataset file (CSV)</label>
          <input type="file" accept=".csv" onChange={e=>{ if(e.target.files[0]) setFile(e.target.files[0]); }} />
          {file && <div className="tiny muted" style={{marginTop:4}}>Loaded: {file.name}</div>}
        </div>
        <div className="row3">
          <div><label>Top K</label>
            <select value={form.top_k} onChange={e=>set('top_k',Number(e.target.value))}>
              <option value={10}>10</option><option value={5}>5</option>
            </select>
          </div>
          <div><label>Top models</label>
            <select value={form.top_models} onChange={e=>set('top_models',Number(e.target.value))}>
              <option value={10}>10</option><option value={5}>5</option>
            </select>
          </div>
          <div><label>Algorithm mode</label>
            <select value={form.algorithm_mode} onChange={e=>set('algorithm_mode',e.target.value)}>
              <option value="auto">Auto</option><option value="explicit">Explicit</option><option value="implicit">Implicit</option>
            </select>
          </div>
        </div>
        <div><label>Optuna trials (-1 = adaptive)</label>
          <input type="number" value={form.n_trials} onChange={e=>set('n_trials',Number(e.target.value))} />
        </div>
        <div className="actions"><button disabled={!file} onClick={train}>Train</button></div>
        <JobStatus data={jobData} label="Training" />
      </div>
    </div>
  );
}

function PageRecommend({ setMsg }) {
  const [loginForm, setLoginForm] = useState({username:'admin',password:'admin123'});
  const [token, setToken] = useState(()=>localStorage.getItem('proactive_token')||'');
  const [models, setModels] = useState([]);
  const [recForm, setRecForm] = useState({user_id:'',top_n:10,algorithm:''});
  const [result, setResult] = useState(null);
  const setL = (k,v) => setLoginForm(p=>({...p,[k]:v}));
  const setR = (k,v) => setRecForm(p=>({...p,[k]:v}));
  const promoted = useMemo(()=>models.find(m=>m.is_promoted||m.promoted),[models]);

  useEffect(()=>{ localStorage.setItem('proactive_token', token||''); },[token]);
  useEffect(()=>{ if(token) loadModels(token); },[]);

  async function login() {
    try {
      const d = await jfetch('/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(loginForm)});
      setToken(d.access_token); setMsg('Login successful.');
      await loadModels(d.access_token);
    } catch(e){ setMsg(e.message); }
  }

  async function loadModels(tok) {
    const auth = tok||token; if (!auth) return;
    try {
      const d = await jfetch('/models',{headers:{Authorization:`Bearer ${auth}`}});
      setModels(d.models||[]);
    } catch(e){ setMsg(e.message); }
  }

  async function recommend() {
    try {
      const payload = {user_id:recForm.user_id,top_n:Number(recForm.top_n),strategy:'single_model',algorithm:recForm.algorithm||promoted?.algorithm||undefined};
      const d = await jfetch('/recommend',{method:'POST',headers:{'Content-Type':'application/json',Authorization:`Bearer ${token}`},body:JSON.stringify(payload)});
      setResult(d);
    } catch(e){ setMsg(e.message); }
  }

  return (
    <div className="stack">
      <div className="panel">
        <h2>Login</h2>
        <div className="stack">
          <div className="row">
            <div><label>Username</label><input value={loginForm.username} onChange={e=>setL('username',e.target.value)} /></div>
            <div><label>Password</label><input type="password" value={loginForm.password} onChange={e=>setL('password',e.target.value)} /></div>
          </div>
          <div className="actions">
            <button onClick={login}>Login</button>
            <button className="sec" onClick={()=>loadModels()}>Refresh models</button>
            {token && <span className="badge ok">Authenticated</span>}
          </div>
        </div>
      </div>

      <div className="panel">
        <h2>Trained models</h2>
        <div className="cards">
          {models.length===0 && <div className="muted tiny">No models yet — train first, then login.</div>}
          {models.slice(0,10).map((m,i)=>(
            <div key={m.model_id||i} className="conn">
              <div>
                <strong>{m.algorithm}</strong>
                {(m.is_promoted||m.promoted) && <span className="badge ok" style={{marginLeft:6}}>Promoted</span>}
                <div className="tiny muted code">{m.model_id}</div>
              </div>
              <div className="tiny muted">score: {m.validation_score??m.score??'n/a'}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="panel">
        <h2>Get recommendations</h2>
        <div className="stack">
          <div className="row3">
            <div><label>User ID</label><input value={recForm.user_id} onChange={e=>setR('user_id',e.target.value)} placeholder="user_123" /></div>
            <div><label>Top N</label>
              <select value={recForm.top_n} onChange={e=>setR('top_n',Number(e.target.value))}>
                <option value={10}>10</option><option value={5}>5</option>
              </select>
            </div>
            <div><label>Algorithm</label>
              <select value={recForm.algorithm} onChange={e=>setR('algorithm',e.target.value)}>
                <option value="">Promoted / default</option>
                {[...new Set(models.map(m=>m.algorithm))].map(a=><option key={a} value={a}>{a}</option>)}
              </select>
            </div>
          </div>
          <div className="actions">
            <button disabled={!token||!recForm.user_id} onClick={recommend}>Get recommendations</button>
          </div>
          <div className="log">{result ? JSON.stringify(result,null,2) : 'Results will appear here.'}</div>
        </div>
      </div>
    </div>
  );
}

// ── App shell ────────────────────────────────────────────────────────────────

export default function App() {
  const [mode, setMode]         = useState(null);   // 'csv' | 'db'
  const [page, setPage]         = useState('mode');
  const [connections, setConnections] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [schemas, setSchemas]   = useState({});
  const [builtFile, setBuiltFile] = useState(null);
  const [msg, setMsg]           = useState('');

  useEffect(()=>{ loadConnections(); },[]);

  async function loadConnections(autoSelectId) {
    try {
      const raw = await jfetch(`${SMART}/connections`);
      const list = raw.map(norm);
      setConnections(list);
      if (autoSelectId) setSelectedIds(p => p.includes(autoSelectId) ? p : [...p, autoSelectId]);
    } catch(e){ setMsg(e.message); }
  }

  async function removeConnection(id) {
    try {
      await jfetch(`${SMART}/connections/${id}`,{method:'DELETE'});
      setSelectedIds(p=>p.filter(x=>x!==id));
      setSchemas(p=>{ const n={...p}; delete n[id]; return n; });
      await loadConnections();
    } catch(e){ setMsg(e.message); }
  }

  async function loadSchema(id) {
    if (schemas[id]) return;
    try {
      const d = await jfetch(`${SMART}/schema/${id}`);
      setSchemas(p=>({...p,[id]:d}));
    } catch(e){ setMsg(e.message); }
  }

  function selectMode(m) {
    setMode(m);
    setPage(m==='csv' ? 'train' : 'connections');
  }

  const PAGES_DB  = ['mode','connections','build','train','recommend'];
  const PAGES_CSV = ['mode','train','recommend'];
  const pages = mode==='csv' ? PAGES_CSV : PAGES_DB;
  const labels = {mode:'Source',connections:'Connections',build:'Build dataset',train:'Train',recommend:'Recommend'};

  return (
    <>
      <style>{css}</style>
      <div className="wrap">
        <div className="hero">
          <div style={{display:'flex',gap:8,flexWrap:'wrap',marginBottom:8}}>
            <span className="badge">Proactive AI</span>
            {mode && <span className="badge ok">{mode==='csv'?'CSV mode':'Database mode'}</span>}
          </div>
          <h1>Proactive AI Unified Dashboard</h1>
          <p>Train recommendation models from a CSV or a live database, then query the promoted model — all from one place.</p>
        </div>

        <div className="steps">
          {pages.map(p=>(
            <button key={p} className={`step-btn${page===p?' active':''}`} onClick={()=>setPage(p)}>
              {labels[p]}
            </button>
          ))}
          {mode && (
            <button className="sec" style={{marginLeft:'auto',padding:'7px 14px',borderRadius:999,border:'1px solid var(--line)',background:'transparent',color:'var(--muted)',fontSize:13,cursor:'pointer'}}
              onClick={()=>{ setMode(null); setPage('mode'); }}>↩ Change source</button>
          )}
        </div>

        <Msg msg={msg} clear={()=>setMsg('')} />

        {page==='mode'        && <PageMode onSelect={selectMode} />}
        {page==='connections' && (
          <PageConnections
            connections={connections.map(norm)}
            selected={selectedIds}
            schemas={schemas}
            onAdded={id=>loadConnections(id)}
            onRemove={removeConnection}
            onToggle={(id,checked)=>setSelectedIds(p=>checked?[...p,id]:p.filter(x=>x!==id))}
            onSchema={loadSchema}
            setMsg={setMsg}
          />
        )}
        {page==='build' && (
          <PageBuild
            selected={selectedIds}
            onBuilt={file=>{ setBuiltFile(file); setPage('train'); setMsg('Dataset ready — proceed to Train.'); }}
            setMsg={setMsg}
          />
        )}
        {page==='train' && (
          <PageTrain
            file={builtFile}
            setFile={setBuiltFile}
            onTrained={()=>setMsg('Training complete — go to Recommend.')}
            setMsg={setMsg}
          />
        )}
        {page==='recommend' && <PageRecommend setMsg={setMsg} />}
      </div>
    </>
  );
}
