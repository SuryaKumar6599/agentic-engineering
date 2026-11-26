import React, { useEffect, useState } from 'react';
import { BarChart3, CheckCircle2, Play, XCircle } from 'lucide-react';
import { apiFetch } from '../auth';
import { EvalResult } from '../types';

export const EvalView: React.FC = () => {
  const [data, setData] = useState<EvalResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const runEval = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await apiFetch('/api/eval/run');
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail || `Access denied (${res.status})`);
      }
      const json: EvalResult = await res.json();
      setData(json);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runEval();
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '1.15rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <BarChart3 size={20} color="var(--accent)" />
            Evaluation Suite: Per-Field Precision &amp; Recall
          </h2>
          <p style={{ margin: '0.25rem 0 0', fontSize: '0.85rem', color: 'var(--ink-muted)' }}>
            Benchmarks extraction, MedDRA PT coding, ICH E2D validity, and seriousness across public openFDA FAERS + synthetic clinical narratives.
          </p>
        </div>
        <button type="button" className="btn btn-primary" onClick={runEval} disabled={loading}>
          <Play size={15} />
          {loading ? 'Running Benchmark...' : 'Re-Run Evaluation'}
        </button>
      </div>

      {error && <div className="badge badge-critical" style={{ whiteSpace: 'normal' }}>{error}</div>}
      {data && (
        <>
          <div className="kpi-grid">
            <div className="card">
              <div className="kpi-label">Macro Precision</div>
              <div className="kpi-val" style={{ color: 'var(--accent)' }}>
                {(data.macro_precision * 100).toFixed(1)}%
              </div>
            </div>
            <div className="card">
              <div className="kpi-label">Macro Recall</div>
              <div className="kpi-val" style={{ color: 'var(--success)' }}>
                {(data.macro_recall * 100).toFixed(1)}%
              </div>
            </div>
            <div className="card">
              <div className="kpi-label">Macro F1-Score</div>
              <div className="kpi-val">{(data.macro_f1 * 100).toFixed(1)}%</div>
            </div>
            <div className="card">
              <div className="kpi-label">Eval Dataset</div>
              <div className="kpi-val">{data.total_cases}</div>
              <div style={{ fontSize: '0.78rem', color: 'var(--ink-muted)' }}>
                {data.openfda_cases} openFDA + {data.synthetic_cases} synthetic
              </div>
            </div>
          </div>

          <div tabIndex={0} role="region" aria-labelledby="eval-field-table" className="table-wrapper stack-mobile">
            <table>
              <caption id="eval-field-table" style={{ textAlign: 'left', padding: '0.75rem 0.85rem', fontWeight: 700 }}>
                Per-Field Extraction &amp; Classification Metrics
              </caption>
              <thead>
                <tr>
                  <th>Field</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1-Score</th>
                  <th>Exact Match</th>
                  <th>TP / FP / FN</th>
                  <th>Support</th>
                </tr>
              </thead>
              <tbody>
                {data.per_field.map((row) => (
                  <tr key={row.field}>
                    <th scope="row">
                      <code>{row.field}</code>
                    </th>
                    <td data-label="Precision">
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ minWidth: 46, fontWeight: 600 }}>{(row.precision * 100).toFixed(1)}%</span>
                        <div className="progress-bar-bg" style={{ width: 80 }}>
                          <div className="progress-bar-fill" style={{ width: `${row.precision * 100}%` }} />
                        </div>
                      </div>
                    </td>
                    <td data-label="Recall">
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ minWidth: 46, fontWeight: 600 }}>{(row.recall * 100).toFixed(1)}%</span>
                        <div className="progress-bar-bg" style={{ width: 80 }}>
                          <div
                            className="progress-bar-fill"
                            style={{ width: `${row.recall * 100}%`, background: 'var(--success)' }}
                          />
                        </div>
                      </div>
                    </td>
                    <td data-label="F1-Score">{(row.f1 * 100).toFixed(1)}%</td>
                    <td data-label="Exact Match">{(row.exact_match * 100).toFixed(1)}%</td>
                    <td data-label="TP / FP / FN">
                      {row.tp} / {row.fp} / {row.fn}
                    </td>
                    <td data-label="Support">{row.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div tabIndex={0} role="region" aria-labelledby="eval-case-table" className="table-wrapper stack-mobile">
            <table>
              <caption id="eval-case-table" style={{ textAlign: 'left', padding: '0.75rem 0.85rem', fontWeight: 700 }}>
                Per-Case Evaluation Breakdown
              </caption>
              <thead>
                <tr>
                  <th>Eval ID</th>
                  <th>Source</th>
                  <th>Validity</th>
                  <th>MedDRA PT</th>
                  <th>Seriousness</th>
                  <th>Predicted vs Gold PTs</th>
                </tr>
              </thead>
              <tbody>
                {data.cases.map((c) => (
                  <tr key={c.id}>
                    <th scope="row">{c.id}</th>
                    <td data-label="Source">{c.source_type}</td>
                    <td data-label="Validity">
                      {c.validity_match ? <CheckCircle2 size={16} color="var(--success)" /> : <XCircle size={16} color="var(--critical)" />}
                    </td>
                    <td data-label="MedDRA PT">
                      {c.meddra_pt_match ? <CheckCircle2 size={16} color="var(--success)" /> : <XCircle size={16} color="var(--critical)" />}
                    </td>
                    <td data-label="Seriousness">
                      {c.seriousness_match ? <CheckCircle2 size={16} color="var(--success)" /> : <XCircle size={16} color="var(--critical)" />}
                    </td>
                    <td data-label="Predicted vs Gold PTs" style={{ fontSize: '0.8rem' }}>
                      <div><strong>Pred:</strong> {c.pred_pts.join(', ') || '—'}</div>
                      <div style={{ color: 'var(--ink-muted)' }}><strong>Gold:</strong> {c.gold_pts.join(', ') || '—'}</div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
};
