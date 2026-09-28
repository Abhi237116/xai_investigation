import { useEffect, useMemo, useState } from "react";
import { Activity, AlertTriangle, ArrowDownRight, ArrowUpRight, Bot, ChevronRight, CircleHelp, Database, FileSearch, GitBranch, LayoutDashboard, Moon, RefreshCw, Search, ShieldCheck, Sun, Waypoints } from "lucide-react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
type Case = { investigation_id: string; transaction_id: string; status: string };
type Report = Record<string, any>;
type Detail = { investigation_id: string; transaction: Record<string, any>; evidence: any[]; latest_report: Report | null; status: string };
const nav = [
  ["Overview", LayoutDashboard], ["Investigations", FileSearch], ["Alerts", AlertTriangle],
  ["Transactions", Database], ["Evidence graph", Waypoints], ["XAI", GitBranch],
  ["Agent activity", Bot], ["Benchmarks", Activity], ["System health", ShieldCheck],
] as const;

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${API}${path}`);
  if (!response.ok) throw new Error(`API request failed (${response.status})`);
  return response.json();
}

export default function App() {
  const [page, setPage] = useState("Overview");
  const [cases, setCases] = useState<Case[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [selected, setSelected] = useState("");
  const [detail, setDetail] = useState<Detail | null>(null);
  const [transactions, setTransactions] = useState<any[]>([]);
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [dark, setDark] = useState(true);
  const [health, setHealth] = useState("checking");
  const [benchmark, setBenchmark] = useState<any>(null);

  async function refresh() {
    setError("");
    try {
      const [nextCases, nextSummary, nextTransactions] = await Promise.all([
        get<Case[]>("/investigations?limit=100"), get("/summary"), get<any[]>("/transactions?limit=100"),
      ]);
      setCases(nextCases); setSummary(nextSummary); setTransactions(nextTransactions); setHealth("online");
      const id = selected || nextCases[0]?.investigation_id;
      if (id) { setSelected(id); setDetail(await get<Detail>(`/investigations/${id}`)); }
    } catch { setHealth("offline"); setError("Investigation API is unavailable. Start the local API to load case records."); }
  }

  useEffect(() => { void refresh(); }, []);
  useEffect(() => { if (page === "Benchmarks") void get<any>("/benchmarks").then(setBenchmark).catch(() => setBenchmark(null)); }, [page]);
  async function selectCase(id: string) {
    setSelected(id); setError("");
    try { setDetail(await get<Detail>(`/investigations/${id}`)); }
    catch { setError("Could not load this investigation."); }
  }
  async function runInvestigation() {
    if (!selected) return;
    setBusy(true); setError("");
    try {
      await fetch(`${API}/investigations/${selected}/run`, { method: "POST" }).then(async r => { if (!r.ok) throw new Error(); return r.json(); });
      await refresh();
    } catch { setError("Investigation run failed. Check API health and model availability."); }
    finally { setBusy(false); }
  }
  const filtered = useMemo(() => cases.filter(item => `${item.investigation_id} ${item.transaction_id}`.toLowerCase().includes(query.toLowerCase())), [cases, query]);
  const tx = detail?.transaction;
  const report = detail?.latest_report;
  const chart = transactions.slice(0, 24).reverse().map(item => ({ time: new Date(item.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }), amount: item.amount }));

  return <div className={dark ? "shell dark" : "shell light"}>
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><ShieldCheck size={19}/></div><div><strong>Sentinel</strong><small>INVESTIGATION DESK</small></div></div>
      <div className="workspace-label">WORKSPACE <span>DEMO</span></div>
      <nav>{nav.map(([name, Icon]) => <button key={name} title={name} className={`nav-item ${page === name ? "active" : ""}`} onClick={() => setPage(name)}><Icon size={17}/><span>{name}</span>{name === "Investigations" && <b>{summary?.investigation_count || 0}</b>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="api-state"><span className={`pulse ${health}`}/><div><strong>API {health}</strong><small>Local evidence store</small></div><button title="Refresh connection" onClick={() => void refresh()}><RefreshCw size={15}/></button></div><div className="profile"><div className="avatar">AR</div><div><strong>Analyst review</strong><small>Human decision required</small></div><CircleHelp size={16}/></div></div>
    </aside>
    <main className="main">
      <header className="topbar"><div className="breadcrumbs">Risk operations <ChevronRight size={14}/><strong>{page}</strong></div><div className="top-actions"><span className="env"><span className="pulse online"/> SYNTHETIC ENVIRONMENT</span><button className="icon-btn" title="Toggle theme" onClick={() => setDark(!dark)}>{dark ? <Sun size={17}/> : <Moon size={17}/>}</button><div className="avatar small">AR</div></div></header>
      <div className="content">
        <div className="page-heading"><div><div className="eyebrow">TUESDAY, SEPTEMBER 29, 2026 <span>·</span> LIVE WORKSPACE</div><h1>{page === "Overview" ? "Investigation overview" : page}</h1><p>Evidence-led review queue <span className="muted-dot">/</span> Human decision support</p></div><button className="primary" onClick={() => void runInvestigation()} disabled={!selected || busy}><RefreshCw size={15} className={busy ? "spin" : ""}/>{busy ? "Running" : "Run investigation"}</button></div>
        {error && <div className="error-banner"><AlertTriangle size={16}/>{error}<button onClick={() => void refresh()}>Retry</button></div>}
        <section className="stats-grid">
          <Stat label="Open investigations" value={summary?.investigation_count ?? "—"} trend="REVIEW QUEUE" icon={<FileSearch size={17}/>} tone="teal"/>
          <Stat label="Awaiting human review" value={summary?.review_required ?? "—"} trend="NO AUTO-DISPOSITION" icon={<AlertTriangle size={17}/>} tone="amber"/>
          <Stat label="Transactions monitored" value={summary?.transaction_count?.toLocaleString() ?? "—"} trend="SYNTHETIC DATA" icon={<Database size={17}/>} tone="blue"/>
          <Stat label="Evidence verification" value={report?.verification_result || "Pending"} trend={report ? "LATEST CASE RUN" : "RUN A CASE"} icon={<ShieldCheck size={17}/>} tone="green"/>
        </section>
        <section className="overview-grid">
          <div className="panel activity-panel"><div className="panel-head"><div><h2>Transaction activity</h2><p>Recent synthetic transaction amounts</p></div><button className="select-like">Recent <ChevronRight size={14}/></button></div><div className="chart-wrap">{chart.length ? <ResponsiveContainer width="100%" height="100%"><AreaChart data={chart}><defs><linearGradient id="amountFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#43c9a5" stopOpacity={0.28}/><stop offset="100%" stopColor="#43c9a5" stopOpacity={0}/></linearGradient></defs><CartesianGrid strokeDasharray="3 5" vertical={false} stroke="var(--line)"/><XAxis dataKey="time" tickLine={false} axisLine={false} tick={{ fill: "var(--muted)", fontSize: 11 }}/><YAxis tickLine={false} axisLine={false} width={55} tick={{ fill: "var(--muted)", fontSize: 11 }} tickFormatter={v => `$${Math.round(v)}`}/><Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)", borderRadius: 6 }}/><Area type="monotone" dataKey="amount" stroke="#43c9a5" strokeWidth={2} fill="url(#amountFill)"/></AreaChart></ResponsiveContainer> : <div className="chart-empty">Transaction activity appears when the API is connected.</div>}</div><div className="chart-foot"><span><i className="legend-dot"/> Transaction amount</span><span>USD <ArrowUpRight size={13}/></span></div></div>
          <div className="panel queue-panel"><div className="panel-head"><div><h2>Review queue</h2><p>Cases requiring analyst attention</p></div><button className="text-btn" onClick={() => setPage("Investigations")}>View all <ArrowUpRight size={14}/></button></div><div className="queue-list">{filtered.slice(0, 5).map((item, index) => <button className={`queue-row ${selected === item.investigation_id ? "chosen" : ""}`} key={item.investigation_id} onClick={() => void selectCase(item.investigation_id)}><span className={`rank rank-${index % 3}`}>{String(index + 1).padStart(2, "0")}</span><span className="queue-meta"><strong>{item.investigation_id}</strong><small>{item.transaction_id} <span>·</span> {item.status}</small></span><ChevronRight size={15}/></button>)}{filtered.length === 0 && <div className="empty-small">No matching investigations</div>}</div><div className="queue-foot"><span><span className="pulse amber"/> Requires human review</span><span>{filtered.length} total</span></div></div>
        </section>
        <section className="workspace-grid">
          <div className="panel case-panel"><div className="panel-head"><div><div className="section-label">INVESTIGATION WORKSPACE</div><h2>{selected || "Select an investigation"}</h2><p>{tx?.transaction_id || "Choose a case from the review queue"} <span>·</span> {detail?.status || ""}</p></div><span className="status-pill"><i/>REQUIRES REVIEW</span></div>
            {tx && <><div className="transaction-strip"><div><small>AMOUNT</small><strong>${Number(tx.amount).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</strong></div><div><small>ACCOUNT</small><strong>{tx.account_id}</strong></div><div><small>BENEFICIARY</small><strong>{tx.beneficiary_id}</strong></div><div><small>EVENT TIME</small><strong>{new Date(tx.timestamp).toLocaleString()}</strong></div></div>
            <div className="evidence-section"><div className="subhead"><h3>Latest investigation report</h3><span>{report ? `${report.observed_evidence?.length || 0} evidence items` : "Not run"}</span></div>{report ? <>
              <div className="report-banner"><div className="report-icon"><ShieldCheck size={17}/></div><div><strong>Evidence verification {report.verification_result}</strong><small>Decision support only. Analyst review remains required.</small></div><span className="model-score">{report.risk_score == null ? "N/A" : `${(report.risk_score * 100).toFixed(1)}%`}<small>MODEL SCORE</small></span></div>
              <div className="evidence-list">{(report.observed_evidence || []).slice(0, 5).map((item: any) => <div className="evidence-row" key={item.evidence_id}><span className={`category-dot ${item.category === "MODEL_OUTPUT" ? "model" : "observed"}`}/><div><span className="evidence-category">{item.category.replaceAll("_", " ")} <small>{item.source}</small></span><p>{item.claim}</p><code>{item.evidence_id}</code></div></div>)}</div>
              <div className="report-columns"><div><h4>SHAP factors</h4>{(report.top_shap_factors || []).slice(0, 3).map((f: any) => <div className="factor" key={f.feature}><span>{f.feature}</span><strong className={f.contribution >= 0 ? "positive" : "negative"}>{f.contribution >= 0 ? "+" : ""}{Number(f.contribution).toFixed(3)}</strong></div>)}{!report.top_shap_factors?.length && <small className="muted">Explanation unavailable</small>}</div><div><h4>Counterfactual simulation</h4><p className="muted">{report.counterfactual?.status === "available" ? `${report.counterfactual.prediction_changed ? "Threshold crossed" : "Threshold unchanged"} in bounded simulation` : "Unavailable for this model run"}</p><span className="simulation-note">SIMULATION ONLY · NOT A REAL-WORLD OUTCOME</span></div></div>
              <div className="review-notice"><ShieldCheck size={16}/><span><strong>Human decision required</strong><small>No automated case disposition is performed by this system.</small></span></div>
            </> : <div className="empty-report"><FileSearch size={22}/><span>Run an investigation to retrieve cited evidence, model output, and explanation.</span><button className="text-btn" onClick={() => void runInvestigation()}>Run now <ArrowUpRight size={14}/></button></div>}</div></>}
          </div>
          <div className="right-stack"><div className="panel network-panel"><div className="panel-head"><div><div className="section-label">RELATIONSHIP SIGNALS</div><h2>Network findings</h2></div><Waypoints size={17}/></div>{report?.network_findings?.length ? report.network_findings.slice(0, 4).map((finding: any) => <div className="network-row" key={finding.evidence_id}><span className="network-node"/><div><strong>{finding.entity_type} shared</strong><small>{finding.entity_id} · {finding.related_account_ids?.length || 0} linked accounts</small></div></div>) : <div className="empty-small">No relationship findings loaded for this case.</div>}</div>
            <div className="panel agent-panel"><div className="panel-head"><div><div className="section-label">INVESTIGATION TRACE</div><h2>Agent activity</h2></div><Bot size={17}/></div><div className="agent-list">{(report?.agent_actions || []).slice(0, 8).map((action: any, index: number) => <div className="agent-row" key={`${action.agent}-${index}`}><span className={`agent-marker ${action.status === "FAIL" ? "failed" : ""}`}/><div><strong>{action.tool || action.action}</strong><small>{action.agent} <span>·</span> {action.status || "complete"}</small></div></div>)}{!report && <div className="empty-small">Run status and approved tool calls will appear here.</div>}</div><div className="agent-foot"><span><span className="pulse online"/> Workflow trace</span><span>{report?.llm_status || "LLM disabled"}</span></div></div></div>
        </section>
        {page === "Investigations" && <section className="panel all-cases"><div className="panel-head"><div><h2>All investigations</h2><p>Search by case or transaction identifier</p></div><label className="search-box"><Search size={15}/><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search investigations"/></label></div><div className="table-head"><span>CASE</span><span>TRANSACTION</span><span>STATUS</span><span>REVIEW</span></div>{filtered.map(item => <button className="table-row" key={item.investigation_id} onClick={() => { void selectCase(item.investigation_id); setPage("Overview"); }}><strong>{item.investigation_id}</strong><span>{item.transaction_id}</span><span>{item.status}</span><span className="status-pill"><i/>Human review</span></button>)}</section>}
        {page === "Alerts" && <section className="panel all-cases"><div className="panel-head"><div><h2>Review alerts</h2><p>Every alert remains subject to analyst review</p></div><span className="status-pill"><i/>{summary?.review_required || 0} OPEN</span></div><div className="table-head"><span>CASE</span><span>TRANSACTION</span><span>STATUS</span><span>REVIEW</span></div>{cases.map(item => <button className="table-row" key={item.investigation_id} onClick={() => { void selectCase(item.investigation_id); setPage("Overview"); }}><strong>{item.investigation_id}</strong><span>{item.transaction_id}</span><span>{item.status}</span><span className="status-pill"><i/>Required</span></button>)}</section>}
        {page === "Transactions" && <section className="panel all-cases"><div className="panel-head"><div><h2>Transaction records</h2><p>Synthetic records; labels are withheld from investigator views</p></div><label className="search-box"><Search size={15}/><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search transaction or account"/></label></div><div className="table-head"><span>TRANSACTION</span><span>ACCOUNT</span><span>AMOUNT</span><span>EVENT TIME</span></div>{transactions.filter(item => `${item.transaction_id} ${item.account_id}`.toLowerCase().includes(query.toLowerCase())).slice(0, 50).map(item => <div className="table-row" key={item.transaction_id}><strong>{item.transaction_id}</strong><span>{item.account_id}</span><span>${Number(item.amount).toFixed(2)}</span><span>{new Date(item.timestamp).toLocaleString()}</span></div>)}</section>}
        {page === "Evidence graph" && <section className="panel all-cases"><div className="panel-head"><div><h2>Relationship evidence</h2><p>Retrieved shared-entity links for the selected case</p></div><Waypoints size={18}/></div>{(report?.network_findings || []).map((item: any) => <div className="table-row" key={item.evidence_id}><strong>{item.entity_type}: {item.entity_id}</strong><span>{item.related_account_ids?.join(", ") || "No linked accounts"}</span><span>{item.category}</span><span>{item.evidence_id}</span></div>)}{!report?.network_findings?.length && <div className="empty-small">Select a case and run an investigation to retrieve graph findings.</div>}</section>}
        {page === "XAI" && <section className="panel all-cases"><div className="panel-head"><div><h2>Model explanation</h2><p>SHAP values come directly from the configured tree model</p></div><GitBranch size={17}/></div><div className="report-banner"><div className="report-icon"><Activity size={17}/></div><div><strong>Model output: {report?.risk_score == null ? "Unavailable" : `${(report.risk_score * 100).toFixed(2)}%`}</strong><small>Version {report?.model_version || "not loaded"} · MODEL_OUTPUT, not a human decision</small></div></div>{(report?.top_shap_factors || []).map((item: any) => <div className="table-row" key={item.feature}><strong>{item.feature}</strong><span>{item.value}</span><span className={item.contribution >= 0 ? "positive" : "negative"}>{Number(item.contribution).toFixed(5)}</span><span>{item.evidence_id}</span></div>)}{!report && <div className="empty-small">Run the selected case to inspect its explanation.</div>}</section>}
        {page === "Agent activity" && <section className="panel all-cases"><div className="panel-head"><div><h2>Agent execution trace</h2><p>Bounded specialist workflow with evidence verification</p></div><Bot size={17}/></div>{(report?.agent_actions || []).map((item: any, index: number) => <div className="table-row" key={`${item.agent}-${index}`}><strong>{item.agent}</strong><span>{item.tool || item.action}</span><span>{item.status || "complete"}</span><span>{item.result_count ?? item.provider ?? ""}</span></div>)}{!report && <div className="empty-small">Run an investigation to inspect the tool trace.</div>}</section>}
        {page === "Benchmarks" && <section className="panel all-cases"><div className="panel-head"><div><h2>Platform benchmark</h2><p>Reproducible synthetic evaluation · not production performance</p></div><Activity size={17}/></div>{benchmark?.ml?.models && Object.entries(benchmark.ml.models).map(([name, metric]: [string, any]) => <div className="table-row" key={name}><strong>{name.replace(/_/g, " ")}</strong><span>PR-AUC {Number(metric.pr_auc).toFixed(3)}</span><span>F1 {Number(metric.f1).toFixed(3)}</span><span>Brier {Number(metric.brier_score).toFixed(4)}</span></div>)}{benchmark?.agents?.metrics && Object.entries(benchmark.agents.metrics).map(([name, value]: [string, any]) => <div className="table-row" key={name}><strong>{name.replace(/_/g, " ")}</strong><span>Mean {Number(value.mean).toFixed(3)}</span><span>p50 {Number(value.p50).toFixed(3)}</span><span>p95 {Number(value.p95).toFixed(3)}</span></div>)}{!benchmark && <div className="empty-small">Run the evaluation suite to produce the benchmark report.</div>}<div className="review-notice"><AlertTriangle size={16}/><span><strong>Limitations</strong><small>{benchmark?.note || "Synthetic benchmark only; no real-world performance claims."} Kafka requires Docker; LLM calls are disabled in the default evaluation.</small></span></div></section>}
        {page === "System health" && <section className="panel all-cases"><div className="panel-head"><div><h2>Service health</h2><p>Local development environment</p></div><ShieldCheck size={17}/></div><div className="table-row"><strong>Investigation API</strong><span>{health}</span><span>GET /health</span><span>Local</span></div><div className="table-row"><strong>Evidence store</strong><span>{summary?.transaction_count ?? "—"} records</span><span>SQLAlchemy</span><span>Ready</span></div><div className="table-row"><strong>LLM provider</strong><span>{report?.llm_status || "disabled"}</span><span>Evidence export opt-in</span><span>Optional</span></div><div className="table-row"><strong>Kafka / Redpanda</strong><span>Not checked</span><span>Docker required</span><span>Optional</span></div><a className="metrics-link" href={`${API}/metrics`} target="_blank" rel="noreferrer">Open Prometheus metrics <ArrowUpRight size={14}/></a></section>}
      </div>
      <footer className="footer"><span>Sentinel Financial Crime Investigation</span><span>Decision-support system <i/> Synthetic data only <i/> v0.1.0</span></footer>
    </main>
  </div>;
}

function Stat({ label, value, trend, icon, tone }: {label: string; value: string | number; trend: string; icon: React.ReactNode; tone: string}) {
  return <div className="stat"><div className={`stat-icon ${tone}`}>{icon}</div><div className="stat-label">{label}</div><div className="stat-value">{value}</div><div className="stat-foot"><span>{trend}</span>{tone === "green" ? <ArrowUpRight size={14}/> : <ArrowDownRight size={14}/>}</div></div>;
}
