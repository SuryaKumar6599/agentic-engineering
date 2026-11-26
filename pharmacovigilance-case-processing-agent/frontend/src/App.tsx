import React, { useEffect, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Clock,
  FileSpreadsheet,
  History,
  Layers,
  Plus,
  Search,
  ShieldAlert,
} from 'lucide-react';
import { apiFetch, EnterpriseUser, getStoredUser, hasPerm } from './auth';
import { AuditTrailEntry, PVCase, STAGE_LABELS, WorkflowStage } from './types';
import { CaseWorkspace } from './components/CaseWorkspace';
import { EvalView } from './components/EvalView';
import { IntakeDialog } from './components/IntakeDialog';
import { RoleLoginModal } from './components/RoleLoginModal';

type NavTab = 'pipeline' | 'evals' | 'audit';

export default function App() {
  const [nav, setNav] = useState<NavTab>('pipeline');
  const [currentUser, setCurrentUser] = useState<EnterpriseUser | null>(() => getStoredUser());
  const [roleModalOpen, setRoleModalOpen] = useState<boolean>(() => !getStoredUser());
  const [cases, setCases] = useState<PVCase[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [stageFilter, setStageFilter] = useState<WorkflowStage | 'ALL'>('ALL');
  const [priorityFilter, setPriorityFilter] = useState<string>('ALL');
  const [searchQ, setSearchQ] = useState('');
  const [intakeOpen, setIntakeOpen] = useState(false);
  const [toast, setToast] = useState('');
  const [globalAudit, setGlobalAudit] = useState<AuditTrailEntry[]>([]);
  const [stats, setStats] = useState<{
    total_cases: number;
    urgent_7day_sla: number;
    closed_submitted: number;
    by_stage: Record<string, number>;
  } | null>(null);

  const loadAll = async () => {
    if (!currentUser) return;
    const [cRes, sRes] = await Promise.all([
      apiFetch(`/api/cases${searchQ ? `?q=${encodeURIComponent(searchQ)}` : ''}`),
      apiFetch('/api/stats'),
    ]);
    if (cRes.status === 401 || sRes.status === 401) {
      setCurrentUser(null);
      setRoleModalOpen(true);
      return;
    }
    if (cRes.ok) setCases(await cRes.json());
    if (sRes.ok) setStats(await sRes.json());
  };

  const loadGlobalAudit = async () => {
    if (!currentUser) return;
    const res = await apiFetch('/api/audit-trail?limit=150');
    if (res.ok) setGlobalAudit(await res.json());
  };

  useEffect(() => {
    if (!currentUser) return;
    const timer = setTimeout(() => {
      loadAll();
    }, 200);
    return () => clearTimeout(timer);
  }, [searchQ, currentUser]);

  useEffect(() => {
    if (nav === 'audit') loadGlobalAudit();
  }, [nav]);

  const filteredCases = cases.filter((c) => {
    if (stageFilter !== 'ALL' && c.current_stage !== stageFilter) return false;
    if (priorityFilter !== 'ALL' && c.priority !== priorityFilter) return false;
    return true;
  });

  const initials = currentUser
    ? currentUser.full_name
        .split(' ')
        .map((p) => p[0])
        .slice(0, 2)
        .join('')
        .toUpperCase()
    : 'PV';

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <ShieldAlert size={20} color="#38bdf8" />
          </div>
          <div>
            <div>PV-Agentic</div>
            <div className="brand-sub">Enterprise 9-Stage Pharmacovigilance Platform</div>
          </div>
        </div>

        <nav className="nav-links" aria-label="Primary navigation">
          <button
            type="button"
            className="nav-btn"
            aria-current={nav === 'pipeline' && !selectedCaseId ? 'page' : undefined}
            onClick={() => {
              setNav('pipeline');
              setSelectedCaseId(null);
            }}
          >
            <Layers size={15} /> Pipeline Queue
          </button>
          {hasPerm(currentUser, 'run_evals') && (
            <button
              type="button"
              className="nav-btn"
              aria-current={nav === 'evals' ? 'page' : undefined}
              onClick={() => {
                setNav('evals');
                setSelectedCaseId(null);
              }}
            >
              <BarChart3 size={15} /> Eval Precision &amp; Recall
            </button>
          )}
          <button
            type="button"
            className="nav-btn"
            aria-current={nav === 'audit' ? 'page' : undefined}
            onClick={() => {
              setNav('audit');
              setSelectedCaseId(null);
            }}
          >
            <History size={15} /> Audit Trail
          </button>
          {hasPerm(currentUser, 'ingest_cases') && (
            <button type="button" className="btn btn-primary" onClick={() => setIntakeOpen(true)}>
              <Plus size={15} /> New Case Intake
            </button>
          )}
          <button type="button" className="user-pill" onClick={() => setRoleModalOpen(true)}>
            <span className="user-avatar">{initials}</span>
            <span>{currentUser ? `${currentUser.full_name} · ${currentUser.role}` : 'Sign In (RBAC)'}</span>
          </button>
        </nav>
      </header>

      <main className="main-content">
        {selectedCaseId ? (
          <CaseWorkspace
            caseId={selectedCaseId}
            currentUser={currentUser}
            onBack={() => setSelectedCaseId(null)}
            onUpdated={loadAll}
          />
        ) : nav === 'evals' ? (
          <EvalView />
        ) : nav === 'audit' ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div className="card">
              <h2 style={{ margin: 0, fontSize: '1.1rem' }}>Immutable Regulatory Audit Trail</h2>
              <p style={{ margin: '0.25rem 0 0', fontSize: '0.85rem', color: 'var(--ink-muted)' }}>
                Append-only history of all automated agent executions, human field edits, and HITL approvals across all cases.
              </p>
            </div>
            <div tabIndex={0} role="region" aria-labelledby="global-audit-cap" className="table-wrapper stack-mobile">
              <table>
                <caption id="global-audit-cap" style={{ display: 'none' }}>Global Audit Trail</caption>
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Case ID</th>
                    <th>Actor</th>
                    <th>Stage</th>
                    <th>Action</th>
                    <th>Rationale</th>
                  </tr>
                </thead>
                <tbody>
                  {globalAudit.map((a, idx) => (
                    <tr key={idx}>
                      <td data-label="Timestamp" style={{ fontSize: '0.78rem', whiteSpace: 'nowrap' }}>
                        {a.timestamp.slice(0, 19).replace('T', ' ')}
                      </td>
                      <th scope="row">
                        <button
                          type="button"
                          className="btn"
                          style={{ padding: '0.2rem 0.5rem', fontSize: '0.78rem' }}
                          onClick={() => {
                            setNav('pipeline');
                            setSelectedCaseId(a.case_id);
                          }}
                        >
                          {a.case_id}
                        </button>
                      </th>
                      <td data-label="Actor">
                        <span className={`badge ${a.actor_type === 'HUMAN' ? 'badge-standard' : 'badge-muted'}`}>
                          {a.actor}
                        </span>
                      </td>
                      <td data-label="Stage" style={{ fontSize: '0.8rem' }}>{a.stage}</td>
                      <td data-label="Action" style={{ fontSize: '0.8rem', fontWeight: 600 }}>{a.action}</td>
                      <td data-label="Rationale" style={{ fontSize: '0.8rem' }}>{a.rationale}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <>
            {/* KPI Cards */}
            <div className="kpi-grid">
              <div className="card kpi-card">
                <div>
                  <div className="kpi-label">Total Safety Cases</div>
                  <div className="kpi-val">{stats?.total_cases ?? cases.length}</div>
                </div>
                <div className="kpi-icon" style={{ background: 'var(--accent-soft)', color: 'var(--accent)' }}>
                  <FileSpreadsheet size={20} />
                </div>
              </div>
              <div className="card kpi-card">
                <div>
                  <div className="kpi-label">7-Day Expedited SLA</div>
                  <div className="kpi-val" style={{ color: 'var(--critical)' }}>
                    {stats?.urgent_7day_sla ?? 0}
                  </div>
                </div>
                <div className="kpi-icon" style={{ background: 'var(--critical-soft)', color: 'var(--critical)' }}>
                  <AlertTriangle size={20} />
                </div>
              </div>
              <div className="card kpi-card">
                <div>
                  <div className="kpi-label">Awaiting HITL Review</div>
                  <div className="kpi-val" style={{ color: 'var(--high)' }}>
                    {cases.filter((c) => c.status.startsWith('AWAITING')).length}
                  </div>
                </div>
                <div className="kpi-icon" style={{ background: 'var(--high-soft)', color: 'var(--high)' }}>
                  <Clock size={20} />
                </div>
              </div>
              <div className="card kpi-card">
                <div>
                  <div className="kpi-label">Closed &amp; Submitted</div>
                  <div className="kpi-val" style={{ color: 'var(--success)' }}>
                    {stats?.closed_submitted ?? 0}
                  </div>
                </div>
                <div className="kpi-icon" style={{ background: 'var(--success-soft)', color: 'var(--success)' }}>
                  <CheckCircle2 size={20} />
                </div>
              </div>
            </div>

            {/* 9-Stage Filter Bar */}
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <span className="kpi-label" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                  <Activity size={14} /> Filter by Workflow Stage (1–9)
                </span>
                <button
                  type="button"
                  className="btn"
                  style={{ padding: '0.25rem 0.6rem', fontSize: '0.78rem' }}
                  onClick={() => setStageFilter('ALL')}
                >
                  Show All ({cases.length})
                </button>
              </div>
              <div className="stage-bar">
                {STAGE_LABELS.map((st) => (
                  <button
                    type="button"
                    key={st.key}
                    className={`stage-pill ${stageFilter === st.key ? 'active' : ''}`}
                    onClick={() => setStageFilter(stageFilter === st.key ? 'ALL' : st.key)}
                  >
                    <span className="stage-pill-name">{st.short}</span>
                    <span className="stage-pill-count">{stats?.by_stage?.[st.key] ?? 0}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Search & Priority Quick-Filter Chips */}
            <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap', justifyContent: 'space-between' }}>
              <div style={{ position: 'relative', flex: 1, minWidth: 240, maxWidth: 420 }}>
                <Search size={15} style={{ position: 'absolute', left: 11, top: 11, color: 'var(--ink-muted)' }} />
                <input
                  className="input"
                  style={{ paddingLeft: '2.1rem' }}
                  placeholder="Search Case ID, Drug, or MedDRA PT..."
                  value={searchQ}
                  onChange={(e) => setSearchQ(e.target.value)}
                />
              </div>
              <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                {['ALL', 'CRITICAL', 'HIGH', 'STANDARD'].map((p) => (
                  <button
                    key={p}
                    type="button"
                    className="chip"
                    aria-pressed={priorityFilter === p}
                    onClick={() => setPriorityFilter(p)}
                  >
                    {p === 'ALL' ? 'All Priorities' : p}
                  </button>
                ))}
              </div>
            </div>

            <div tabIndex={0} role="region" aria-labelledby="case-queue-cap" className="table-wrapper stack-mobile">
              <table>
                <caption id="case-queue-cap" style={{ display: 'none' }}>Pharmacovigilance Case Queue</caption>
                <thead>
                  <tr>
                    <th>Case ID</th>
                    <th>Priority / SLA</th>
                    <th>Current Stage</th>
                    <th>Status</th>
                    <th>Suspect Drug(s)</th>
                    <th>MedDRA PT(s)</th>
                    <th>Causality</th>
                    <th>Duplicate</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredCases.map((c) => (
                    <tr
                      key={c.case_id}
                      style={{ cursor: 'pointer' }}
                      onClick={() => setSelectedCaseId(c.case_id)}
                    >
                      <th scope="row">
                        <span style={{ color: 'var(--accent)', textDecoration: 'underline' }}>{c.case_id}</span>
                        {c.safetyreportid && (
                          <div style={{ fontSize: '0.72rem', color: 'var(--ink-muted)', fontWeight: 400 }}>
                            FAERS #{c.safetyreportid}
                          </div>
                        )}
                      </th>
                      <td data-label="Priority / SLA">
                        <span className={`badge ${c.priority === 'CRITICAL' ? 'badge-critical' : c.priority === 'HIGH' ? 'badge-high' : 'badge-standard'}`}>
                          {c.priority} ({c.regulatory_deadline_days}d)
                        </span>
                      </td>
                      <td data-label="Current Stage" style={{ fontSize: '0.82rem' }}>{c.current_stage}</td>
                      <td data-label="Status">
                        <span className={`badge ${c.status === 'CLOSED_SUBMITTED' ? 'badge-success' : c.status === 'INVALID_CASE' || c.status === 'REWORK_REQUIRED' ? 'badge-critical' : 'badge-muted'}`}>
                          {c.status}
                        </span>
                      </td>
                      <td data-label="Suspect Drug(s)" style={{ fontSize: '0.82rem' }}>
                        {c.structured_data.suspect_drugs.map((d) => d.name).join(', ') || '—'}
                      </td>
                      <td data-label="MedDRA PT(s)" style={{ fontSize: '0.82rem' }}>
                        {c.meddra_codings.map((m) => m.pt).slice(0, 3).join(', ')}
                        {c.meddra_codings.length > 3 ? ` +${c.meddra_codings.length - 3}` : ''}
                      </td>
                      <td data-label="Causality" style={{ fontSize: '0.82rem' }}>{c.causality.who_umc_category}</td>
                      <td data-label="Duplicate">
                        <span className="badge badge-muted">{c.duplicate_status}</span>
                      </td>
                    </tr>
                  ))}
                  {filteredCases.length === 0 && (
                    <tr>
                      <td colSpan={8} style={{ textAlign: 'center', padding: '2rem', color: 'var(--ink-muted)' }}>
                        No cases match the current filter. Click "New Case Intake" to ingest public openFDA or narrative reports.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </>
        )}
      </main>

      <IntakeDialog
        open={intakeOpen}
        onClose={() => setIntakeOpen(false)}
        onIngested={(newCase) => {
          loadAll();
          setToast('Case(s) ingested and processed through multi-agent pipeline.');
          setTimeout(() => setToast(''), 3500);
          if (newCase) setSelectedCaseId(newCase.case_id);
        }}
      />

      {toast && (
        <div className="toast" role="status">
          <CheckCircle2 size={16} color="#34d399" />
          {toast}
        </div>
      )}

      <RoleLoginModal
        open={roleModalOpen}
        onClose={() => setRoleModalOpen(false)}
        onAuthenticated={(user) => {
          setCurrentUser(user);
          if (nav === 'evals' && !hasPerm(user, 'run_evals')) setNav('pipeline');
          loadAll();
        }}
        allowCancel={Boolean(currentUser)}
      />
    </div>
  );
}
