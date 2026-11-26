import React, { useEffect, useState } from 'react';
import {
  ArrowLeft,
  CheckCircle2,
  ClipboardCheck,
  Download,
  FileWarning,
  History,
  Lock,
  RefreshCw,
  RotateCcw,
  Save,
  ShieldCheck,
  Stethoscope,
  UserCheck,
} from 'lucide-react';
import { apiFetch, EnterpriseUser, hasPerm } from '../auth';
import { AuditTrailEntry, MedDRACoding, PVCase, STAGE_LABELS } from '../types';

interface CaseWorkspaceProps {
  caseId: string;
  currentUser: EnterpriseUser | null;
  onBack: () => void;
  onUpdated: () => void;
}

export const CaseWorkspace: React.FC<CaseWorkspaceProps> = ({ caseId, currentUser, onBack, onUpdated }) => {
  const [pvCase, setPvCase] = useState<PVCase | null>(null);
  const [audit, setAudit] = useState<AuditTrailEntry[]>([]);
  const [tab, setTab] = useState<'structured' | 'assessments' | 'reviews' | 'traces' | 'audit'>('structured');

  // Editable states
  const [narrative, setNarrative] = useState('');
  const [codings, setCodings] = useState<MedDRACoding[]>([]);
  const [whoCategory, setWhoCategory] = useState('');
  const [isSerious, setIsSerious] = useState(false);

  // HITL Gate states
  const [comments, setComments] = useState('');
  const [saving, setSaving] = useState(false);
  const [hitlError, setHitlError] = useState('');
  const [toast, setToast] = useState('');

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(''), 3200);
  };

  const loadDetail = async () => {
    const res = await apiFetch(`/api/cases/${encodeURIComponent(caseId)}`);
    if (!res.ok) return;
    const data = await res.json();
    const c: PVCase = data.case;
    setPvCase(c);
    setAudit(data.audit_trail);
    setNarrative(c.narrative_text);
    setCodings(c.meddra_codings);
    setWhoCategory(c.causality.who_umc_category);
    setIsSerious(c.seriousness.is_serious);
  };

  useEffect(() => {
    loadDetail();
  }, [caseId]);

  if (!pvCase) {
    return <div className="card">Loading case {caseId}...</div>;
  }

  const currentStageIdx = STAGE_LABELS.findIndex((s) => s.key === pvCase.current_stage);
  const currentStageInfo = STAGE_LABELS[currentStageIdx] || STAGE_LABELS[0];

  const canEditFields = hasPerm(currentUser, 'edit_case_fields');

  const stagePermMap: Record<string, string> = {
    '5_CASE_PROCESSING': 'submit_to_qc',
    '6_QUALITY_CONTROL': 'approve_qc',
    '7_MEDICAL_REVIEW': 'approve_medical',
    '8_PROJECT_LEAD_REVIEW': 'approve_lead',
  };
  const requiredPerm = stagePermMap[pvCase.current_stage] || 'approve_qc';
  const canApproveStage = hasPerm(currentUser, requiredPerm);
  const canReturn = hasPerm(currentUser, 'return_for_correction');
  const canReject = hasPerm(currentUser, 'reject_invalid');

  const handleSaveEdits = async () => {
    setSaving(true);
    setHitlError('');
    try {
      const res = await apiFetch(`/api/cases/${encodeURIComponent(caseId)}/fields`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          actor: currentUser?.full_name || 'Case Processor',
          rationale: 'Human-in-the-loop field / MedDRA / narrative edit',
          meddra_codings: codings,
          narrative_text: narrative,
          causality: { ...pvCase.causality, who_umc_category: whoCategory },
          seriousness: { ...pvCase.seriousness, is_serious: isSerious },
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail || `Save failed (${res.status})`);
      }
      await loadDetail();
      onUpdated();
      showToast('Field edits saved and recorded in audit trail.');
    } catch (err) {
      setHitlError(String(err));
    } finally {
      setSaving(false);
    }
  };

  const handleReassess = async () => {
    setSaving(true);
    setHitlError('');
    try {
      const res = await apiFetch(`/api/cases/${encodeURIComponent(caseId)}/reassess`, { method: 'POST' });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail || `Reassess failed (${res.status})`);
      }
      await loadDetail();
      onUpdated();
      showToast('Field edits saved and recorded in audit trail.');
    } catch (err) {
      setHitlError(String(err));
    } finally {
      setSaving(false);
    }
  };

  const handleDownloadE2B = async () => {
    const res = await apiFetch(`/api/cases/${encodeURIComponent(caseId)}/export-e2b`);
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${caseId}-e2b.xml`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleHITL = async (decision: 'APPROVE' | 'RETURN_FOR_CORRECTION' | 'REJECT_INVALID' | 'CLOSE_FINALIZE') => {
    setSaving(true);
    setHitlError('');
    try {
      const res = await apiFetch(`/api/cases/${encodeURIComponent(caseId)}/hitl`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reviewer_name: currentUser?.full_name || 'Reviewer',
          reviewer_role: currentUser?.role || 'PV_REVIEWER',
          decision,
          comments,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail || `Action denied (${res.status})`);
      }
      setComments('');
      await loadDetail();
      onUpdated();
      showToast(`HITL decision (${decision}) recorded.`);
      showToast('Field edits saved and recorded in audit trail.');
    } catch (err) {
      setHitlError(String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem' }}>
        <button type="button" className="btn" onClick={onBack}>
          <ArrowLeft size={16} /> Back to Case Queue
        </button>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <span className={`badge ${pvCase.priority === 'CRITICAL' ? 'badge-critical' : pvCase.priority === 'HIGH' ? 'badge-high' : 'badge-standard'}`}>
            {pvCase.priority} • {pvCase.triage.sla_label}
          </span>
          <span className="badge badge-muted">{pvCase.status}</span>
          {pvCase.regulatory_submission_id && (
            <span className="badge badge-success">E2B ID: {pvCase.regulatory_submission_id}</span>
          )}
        </div>
      </div>

      {/* 9-Stage Visual Pipeline Stepper */}
      <div className="card">
        <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--ink-muted)', marginBottom: '0.5rem' }}>
          9-STAGE PHARMACOVIGILANCE WORKFLOW — {pvCase.case_id}
        </div>
        <div className="stage-bar">
          {STAGE_LABELS.map((st, idx) => {
            const isCurrent = st.key === pvCase.current_stage;
            const isDone = idx < currentStageIdx || pvCase.status === 'CLOSED_SUBMITTED';
            return (
              <div
                key={st.key}
                className={`stage-pill ${isCurrent ? 'active' : isDone ? 'done' : ''}`}
                style={{ cursor: 'default' }}
              >
                <span className="stage-pill-name" style={{ color: isDone ? 'var(--success)' : isCurrent ? 'var(--accent)' : undefined }}>
                  {isDone ? '✓ ' : ''}{st.short}
                </span>
                <span style={{ fontSize: '0.72rem', color: 'var(--ink-muted)' }}>{st.role}</span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="workspace-grid">
        {/* Left Main Tabs */}
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div role="tablist" aria-label="Case workspace sections" className="tablist" style={{ padding: '0 0.75rem' }}>
            <button type="button" role="tab" aria-selected={tab === 'structured'} onClick={() => setTab('structured')}>
              Structured &amp; MedDRA
            </button>
            <button type="button" role="tab" aria-selected={tab === 'assessments'} onClick={() => setTab('assessments')}>
              Assessments &amp; Narrative
            </button>
            <button type="button" role="tab" aria-selected={tab === 'reviews'} onClick={() => setTab('reviews')}>
              QC &amp; Medical Review
            </button>
            <button type="button" role="tab" aria-selected={tab === 'traces'} onClick={() => setTab('traces')}>
              Agent Trace ({pvCase.agent_traces.length})
            </button>
            <button type="button" role="tab" aria-selected={tab === 'audit'} onClick={() => setTab('audit')}>
              Audit Trail ({audit.length})
            </button>
          </div>

          <div style={{ padding: '1.15rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {tab === 'structured' && (
              <>
                <div className="field-grid">
                  <div className="card" style={{ background: 'var(--surface-muted)' }}>
                    <div className="kpi-label">Stage 2: Validity (ICH E2D)</div>
                    <div style={{ marginTop: '0.35rem', fontWeight: 700, color: pvCase.validity.is_valid ? 'var(--success)' : 'var(--critical)' }}>
                      {pvCase.validity.is_valid ? 'Valid Case (4/4 Criteria)' : `Invalid: ${pvCase.validity.missing_criteria.join(', ')}`}
                    </div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--ink-muted)', marginTop: '0.25rem' }}>
                      Patient: {pvCase.validity.identifiable_patient ? '✓' : '✗'} | Reporter: {pvCase.validity.identifiable_reporter ? '✓' : '✗'} | Drug: {pvCase.validity.suspect_product ? '✓' : '✗'} | AE: {pvCase.validity.adverse_event ? '✓' : '✗'}
                    </div>
                  </div>

                  <div className="card" style={{ background: 'var(--surface-muted)' }}>
                    <div className="kpi-label">Stage 4: Duplicate Check</div>
                    <div style={{ marginTop: '0.35rem', fontWeight: 700 }}>{pvCase.duplicate_check.status}</div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--ink-muted)', marginTop: '0.25rem' }}>
                      {pvCase.duplicate_check.match_reasons.join('; ')}
                    </div>
                  </div>
                </div>

                <div className="field-grid">
                  <div>
                    <div className="kpi-label">Patient</div>
                    <div>Age: {pvCase.structured_data.patient_age || 'Unknown'} | Sex: {pvCase.structured_data.patient_sex || 'Unknown'}</div>
                  </div>
                  <div>
                    <div className="kpi-label">Reporter &amp; Country</div>
                    <div>{pvCase.structured_data.reporter_qualification || 'Unknown'} ({pvCase.structured_data.country || 'N/A'})</div>
                  </div>
                  <div>
                    <div className="kpi-label">Outcome</div>
                    <div>{pvCase.structured_data.reaction_outcome || 'Unknown'}</div>
                  </div>
                </div>

                <div>
                  <div className="kpi-label" style={{ marginBottom: '0.4rem' }}>Suspect &amp; Concomitant Drugs</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                    {pvCase.structured_data.suspect_drugs.map((d, i) => (
                      <span key={i} className="badge badge-critical">
                        Suspect: {d.name} {d.dose ? `(${d.dose})` : ''}
                      </span>
                    ))}
                    {pvCase.structured_data.concomitant_drugs.map((d, i) => (
                      <span key={i} className="badge badge-muted">
                        Concomitant: {d.name}
                      </span>
                    ))}
                  </div>
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                    <span className="kpi-label">Stage 5: MedDRA Coding Suggestions (Editable)</span>
                    <div style={{ display: 'flex', gap: '0.4rem' }}>
                      {canEditFields && (
                        <button type="button" className="btn" onClick={handleReassess} disabled={saving}>
                          <RefreshCw size={14} /> Re-run Clinical &amp; QC
                        </button>
                      )}
                      <button type="button" className="btn btn-primary" onClick={handleSaveEdits} disabled={saving || !canEditFields}>
                        <Save size={14} /> {canEditFields ? 'Save Field Edits' : 'Read-Only (Requires Case Processor)'}
                      </button>
                    </div>
                  </div>
                  <div className="table-wrapper">
                    <table>
                      <thead>
                        <tr>
                          <th>Verbatim</th>
                          <th>Preferred Term (PT)</th>
                          <th>System Organ Class (SOC)</th>
                          <th>Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {codings.map((c, idx) => (
                          <tr key={idx}>
                            <td>{c.verbatim_term}</td>
                            <td>
                              <input
                                className="input"
                                value={c.pt}
                                disabled={!canEditFields}
                                onChange={(e) => {
                                  const next = [...codings];
                                  next[idx] = { ...c, pt: e.target.value };
                                  setCodings(next);
                                }}
                              />
                            </td>
                            <td style={{ fontSize: '0.8rem' }}>{c.soc}</td>
                            <td>
                              <span className={`badge ${c.confidence >= 0.85 ? 'badge-success' : 'badge-high'}`}>
                                {(c.confidence * 100).toFixed(0)}%
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <details className="card" style={{ background: 'var(--surface-muted)', padding: '0.75rem 1rem' }}>
                  <summary style={{ cursor: 'pointer', fontWeight: 600, fontSize: '0.84rem', color: 'var(--ink-muted)' }}>
                    View Raw Source Report ({pvCase.source_type})
                  </summary>
                  <pre className="raw-json">{JSON.stringify(pvCase.raw_report, null, 2)}</pre>
                </details>
              </>
            )}

            {tab === 'assessments' && (
              <>
                <div className="field-grid">
                  <div className="card" style={{ background: 'var(--surface-muted)' }}>
                    <div className="kpi-label">Seriousness Assessment (ICH E2A)</div>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.4rem', fontWeight: 600 }}>
                      <input type="checkbox" checked={isSerious} onChange={(e) => setIsSerious(e.target.checked)} />
                      {isSerious ? 'Serious Case' : 'Non-Serious Case'}
                    </label>
                    <div style={{ fontSize: '0.8rem', color: 'var(--ink-muted)', marginTop: '0.35rem' }}>
                      {pvCase.seriousness.rationale}
                    </div>
                  </div>

                  <div className="card" style={{ background: 'var(--surface-muted)' }}>
                    <div className="kpi-label">Causality (WHO-UMC / Naranjo)</div>
                    <select className="input" style={{ marginTop: '0.35rem' }} value={whoCategory} onChange={(e) => setWhoCategory(e.target.value)}>
                      <option value="Certain">Certain</option>
                      <option value="Probable/Likely">Probable/Likely</option>
                      <option value="Possible">Possible</option>
                      <option value="Unlikely">Unlikely</option>
                      <option value="Unassessable">Unassessable</option>
                    </select>
                    <div style={{ fontSize: '0.8rem', color: 'var(--ink-muted)', marginTop: '0.35rem' }}>
                      Naranjo Score: {pvCase.causality.naranjo_score} ({pvCase.causality.naranjo_category})
                    </div>
                  </div>

                  <div className="card" style={{ background: 'var(--surface-muted)' }}>
                    <div className="kpi-label">Expectedness / Listedness</div>
                    <div style={{ fontWeight: 700, marginTop: '0.35rem' }}>{pvCase.listedness.status}</div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--ink-muted)', marginTop: '0.25rem' }}>
                      {pvCase.listedness.rationale}
                    </div>
                  </div>
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                    <span className="kpi-label">Drafted Clinical Narrative (ICH E2B)</span>
                    <button type="button" className="btn btn-primary" onClick={handleSaveEdits} disabled={saving || !canEditFields}>
                      <Save size={14} /> {canEditFields ? 'Save Assessments & Narrative' : 'Read-Only'}
                    </button>
                  </div>
                  <textarea rows={6} value={narrative} onChange={(e) => setNarrative(e.target.value)} />
                </div>
              </>
            )}

            {tab === 'reviews' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div className="card" style={{ background: 'var(--surface-muted)' }}>
                  <h3 style={{ margin: '0 0 0.5rem', fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <ClipboardCheck size={17} color="var(--accent)" /> Stage 6: Automated Quality Control (QC) Checklist
                  </h3>
                  {pvCase.qc_checklist ? (
                    <div style={{ fontSize: '0.85rem', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                      <div>Status: <strong>{pvCase.qc_checklist.passed ? 'PASSED' : 'ISSUES FLAGGED'}</strong></div>
                      <div>
                        Data Entry: {pvCase.qc_checklist.data_entry_accurate ? '✓' : '✗'} | Completeness: {pvCase.qc_checklist.completeness_ok ? '✓' : '✗'} | Coding: {pvCase.qc_checklist.coding_accurate ? '✓' : '✗'} | Assessments: {pvCase.qc_checklist.assessments_consistent ? '✓' : '✗'} | Narrative: {pvCase.qc_checklist.narrative_consistent ? '✓' : '✗'} | SOP: {pvCase.qc_checklist.sop_compliant ? '✓' : '✗'}
                      </div>
                      {pvCase.qc_checklist.issues.length > 0 && (
                        <ul style={{ margin: '0.25rem 0 0', color: 'var(--critical)' }}>
                          {pvCase.qc_checklist.issues.map((iss, i) => <li key={i}>{iss}</li>)}
                        </ul>
                      )}
                    </div>
                  ) : (
                    <div style={{ fontSize: '0.85rem', color: 'var(--ink-muted)' }}>QC not yet executed.</div>
                  )}
                </div>

                <div className="card" style={{ background: 'var(--surface-muted)' }}>
                  <h3 style={{ margin: '0 0 0.5rem', fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <Stethoscope size={17} color="var(--accent)" /> Stage 7: Medical Review (MBBS/MD Pre-Check)
                  </h3>
                  {pvCase.medical_review ? (
                    <div style={{ fontSize: '0.85rem' }}>
                      <div><strong>Clinically Sound:</strong> {pvCase.medical_review.clinically_sound ? 'Yes' : 'Flagged'}</div>
                      <div style={{ marginTop: '0.25rem' }}>{pvCase.medical_review.clinical_notes}</div>
                    </div>
                  ) : (
                    <div style={{ fontSize: '0.85rem', color: 'var(--ink-muted)' }}>Pending QC approval before Medical Review runs.</div>
                  )}
                </div>
              </div>
            )}

            {tab === 'traces' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
                {pvCase.agent_traces.map((tr, i) => (
                  <div key={i} className="card" style={{ background: 'var(--surface-muted)', padding: '0.75rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', fontWeight: 700, gap: '0.5rem', flexWrap: 'wrap' }}>
                      <span>{tr.agent_name} — {tr.stage}</span>
                      <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
                        {typeof tr.details?.llm_model === 'string' && (
                          <span className="badge badge-standard">{tr.details.llm_model}</span>
                        )}
                        <span className="badge badge-success">{tr.status} ({tr.duration_ms}ms)</span>
                      </div>
                    </div>
                    <div style={{ fontSize: '0.84rem', marginTop: '0.25rem' }}>{tr.summary}</div>
                  </div>
                ))}
              </div>
            )}

            {tab === 'audit' && (
              <div className="table-wrapper">
                <table>
                  <thead>
                    <tr>
                      <th>Timestamp</th>
                      <th>Actor</th>
                      <th>Action</th>
                      <th>Field / Transition</th>
                      <th>Rationale</th>
                    </tr>
                  </thead>
                  <tbody>
                    {audit.map((a, idx) => (
                      <tr key={idx}>
                        <td style={{ fontSize: '0.75rem', whiteSpace: 'nowrap' }}>{a.timestamp.slice(0, 19).replace('T', ' ')}</td>
                        <td>
                          <span className={`badge ${a.actor_type === 'HUMAN' ? 'badge-standard' : 'badge-muted'}`}>
                            {a.actor}
                          </span>
                        </td>
                        <td style={{ fontSize: '0.8rem', fontWeight: 600 }}>{a.action}</td>
                        <td style={{ fontSize: '0.78rem' }}>
                          {a.field_name ? `${a.field_name}: ${a.old_value || '—'} → ${a.new_value || '—'}` : a.stage}
                        </td>
                        <td style={{ fontSize: '0.78rem' }}>{a.rationale}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        {/* Right Panel: Human-in-the-Loop (HITL) Approval Gate */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
          <h3 style={{ margin: 0, fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
            <UserCheck size={18} color="var(--accent)" />
            Human-in-the-Loop Gate
          </h3>
          <div style={{ fontSize: '0.84rem', color: 'var(--ink-muted)' }}>
            Current Stage: <strong>{currentStageInfo.full}</strong>
            <br />
            Required Role: <strong>{currentStageInfo.role}</strong>
            <br />
            Signed in as: <strong>{currentUser?.full_name || 'Guest'}</strong> ({currentUser?.role || 'NONE'})
          </div>

          {hitlError && <div className="badge badge-critical" style={{ whiteSpace: 'normal' }}>{hitlError}</div>}

          {!canApproveStage && pvCase.status !== 'CLOSED_SUBMITTED' && pvCase.status !== 'INVALID_CASE' && (
            <div className="badge badge-high" style={{ whiteSpace: 'normal', padding: '0.5rem 0.65rem' }}>
              <Lock size={14} style={{ flexShrink: 0 }} />
              Separation of Duties: Stage approval requires {currentStageInfo.role} or ADMIN.
            </div>
          )}

          <label>
            <span className="kpi-label">Review Comments / Audit Rationale</span>
            <textarea
              rows={3}
              value={comments}
              onChange={(e) => setComments(e.target.value)}
              placeholder="Enter QC, MD, or Lead approval notes..."
            />
          </label>

          {pvCase.status !== 'CLOSED_SUBMITTED' && pvCase.status !== 'INVALID_CASE' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <button type="button" className="btn btn-success" onClick={() => handleHITL('APPROVE')} disabled={saving || !canApproveStage}>
                <CheckCircle2 size={16} />
                {pvCase.current_stage === '8_PROJECT_LEAD_REVIEW'
                  ? 'Final Approve & Close Case (Stage 9)'
                  : 'Approve & Advance Stage'}
              </button>
              <button type="button" className="btn btn-warn" onClick={() => handleHITL('RETURN_FOR_CORRECTION')} disabled={saving || !canReturn}>
                <RotateCcw size={16} /> Return for Correction (Stage 5)
              </button>
              <button type="button" className="btn btn-danger" onClick={() => handleHITL('REJECT_INVALID')} disabled={saving || !canReject}>
                <FileWarning size={16} /> Mark Invalid Case
              </button>
            </div>
          )}

          {pvCase.status === 'CLOSED_SUBMITTED' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div className="badge badge-success" style={{ padding: '0.6rem', justifyContent: 'center' }}>
                <ShieldCheck size={16} /> Case Finalized &amp; Audit Trail Retained
              </div>
              <button type="button" className="btn btn-primary" onClick={handleDownloadE2B}>
                <Download size={15} /> Download E2B(R3) XML
              </button>
            </div>
          )}

          <div style={{ fontSize: '0.75rem', color: 'var(--ink-muted)', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <History size={13} /> Every HITL action is permanently recorded in the immutable audit trail.
          </div>
        </div>
      </div>
      {toast && (
        <div className="toast" role="status">
          <CheckCircle2 size={16} color="#34d399" />
          {toast}
        </div>
      )}
    </div>
  );
};
