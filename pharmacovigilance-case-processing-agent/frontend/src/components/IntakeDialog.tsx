import React, { useEffect, useRef, useState } from 'react';
import { BookOpen, Cpu, Database, FileText, FlaskConical, X } from 'lucide-react';
import { apiFetch } from '../auth';
import { PVCase } from '../types';

interface IntakeDialogProps {
  open: boolean;
  onClose: () => void;
  onIngested: (newCase?: PVCase) => void;
}

type IntakeMode = 'openfda' | 'device' | 'pubmed' | 'trials' | 'text';

const SAMPLE_TEXT =
  'Physician Dr. Sarah Chen reported a 67 year-old male patient who developed acute kidney injury and severe gastrointestinal bleeding 5 days after starting Ibuprofen 800 mg TID. Concomitant medications: Lisinopril 10 mg. Patient was hospitalized for 3 days and improved after stopping Ibuprofen.';

export const IntakeDialog: React.FC<IntakeDialogProps> = ({ open, onClose, onIngested }) => {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [mode, setMode] = useState<IntakeMode>('openfda');
  const [drugName, setDrugName] = useState('aspirin');
  const [deviceName, setDeviceName] = useState('insulin pump');
  const [pubmedQuery, setPubmedQuery] = useState('pembrolizumab hepatotoxicity case report');
  const [trialQuery, setTrialQuery] = useState('semaglutide');
  const [limit, setLimit] = useState(2);
  const [sourceType, setSourceType] = useState('Spontaneous (HCP)');
  const [rawText, setRawText] = useState(SAMPLE_TEXT);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const dlg = dialogRef.current;
    if (!dlg) return;
    if (open && !dlg.open) {
      dlg.showModal();
    } else if (!open && dlg.open) {
      dlg.close();
    }
  }, [open]);

  const runRemoteIngest = async (url: string) => {
    setLoading(true);
    setError('');
    try {
      const res = await apiFetch(url, { method: 'POST' });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail || `Request failed (${res.status})`);
      }
      onIngested();
      onClose();
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (mode === 'openfda') {
      await runRemoteIngest(`/api/cases/ingest-openfda?drug_name=${encodeURIComponent(drugName)}&limit=${limit}`);
    } else if (mode === 'device') {
      await runRemoteIngest(`/api/cases/ingest-openfda-device?device_name=${encodeURIComponent(deviceName)}&limit=${limit}`);
    } else if (mode === 'pubmed') {
      await runRemoteIngest(`/api/cases/ingest-pubmed?query=${encodeURIComponent(pubmedQuery)}&limit=${limit}`);
    } else if (mode === 'trials') {
      await runRemoteIngest(`/api/cases/ingest-clinicaltrials?query=${encodeURIComponent(trialQuery)}&limit=${limit}`);
    } else {
      setLoading(true);
      setError('');
      try {
        const res = await apiFetch('/api/cases/ingest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source_type: sourceType, raw_text: rawText }),
        });
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail || `Request failed (${res.status})`);
        }
        const created: PVCase = await res.json();
        onIngested(created);
        onClose();
      } catch (err) {
        setError(String(err));
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <dialog
      ref={dialogRef}
      className="pv-dialog"
      aria-labelledby="intake-dialog-title"
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
    >
      <div className="dialog-head">
        <h1 id="intake-dialog-title">Stage 1: Multi-Source Case Intake</h1>
        <button type="button" className="btn" onClick={onClose} aria-label="Close dialog">
          <X size={16} />
        </button>
      </div>

      <div className="dialog-body">
        <div role="tablist" aria-label="Intake source mode" className="tablist">
          <button type="button" role="tab" aria-selected={mode === 'openfda'} onClick={() => setMode('openfda')}>
            <Database size={14} style={{ marginRight: 5, verticalAlign: 'text-bottom' }} />
            FAERS (Drug)
          </button>
          <button type="button" role="tab" aria-selected={mode === 'device'} onClick={() => setMode('device')}>
            <Cpu size={14} style={{ marginRight: 5, verticalAlign: 'text-bottom' }} />
            MAUDE (Device)
          </button>
          <button type="button" role="tab" aria-selected={mode === 'pubmed'} onClick={() => setMode('pubmed')}>
            <BookOpen size={14} style={{ marginRight: 5, verticalAlign: 'text-bottom' }} />
            PubMed
          </button>
          <button type="button" role="tab" aria-selected={mode === 'trials'} onClick={() => setMode('trials')}>
            <FlaskConical size={14} style={{ marginRight: 5, verticalAlign: 'text-bottom' }} />
            ClinicalTrials
          </button>
          <button type="button" role="tab" aria-selected={mode === 'text'} onClick={() => setMode('text')}>
            <FileText size={14} style={{ marginRight: 5, verticalAlign: 'text-bottom' }} />
            Narrative
          </button>
        </div>

        {error && <div className="badge badge-critical" style={{ whiteSpace: 'normal' }}>{error}</div>}

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
          {mode === 'openfda' && (
            <>
              <p style={{ margin: 0, fontSize: '0.84rem', color: 'var(--ink-muted)' }}>
                Fetches public, de-identified drug adverse event reports from openFDA FAERS.
              </p>
              <div className="field-grid">
                <label>
                  <span className="kpi-label">Suspect Drug Name</span>
                  <input className="input" value={drugName} onChange={(e) => setDrugName(e.target.value)} required autoFocus />
                </label>
                <label>
                  <span className="kpi-label">Report Count (1–10)</span>
                  <input className="input" type="number" min={1} max={10} value={limit} onChange={(e) => setLimit(Number(e.target.value))} />
                </label>
              </div>
            </>
          )}

          {mode === 'device' && (
            <>
              <p style={{ margin: 0, fontSize: '0.84rem', color: 'var(--ink-muted)' }}>
                Fetches public medical device adverse event reports from openFDA MAUDE (e.g., insulin pump, pacemaker, stent).
              </p>
              <div className="field-grid">
                <label>
                  <span className="kpi-label">Device Generic Name</span>
                  <input className="input" value={deviceName} onChange={(e) => setDeviceName(e.target.value)} required autoFocus />
                </label>
                <label>
                  <span className="kpi-label">Report Count (1–10)</span>
                  <input className="input" type="number" min={1} max={10} value={limit} onChange={(e) => setLimit(Number(e.target.value))} />
                </label>
              </div>
            </>
          )}

          {mode === 'pubmed' && (
            <>
              <p style={{ margin: 0, fontSize: '0.84rem', color: 'var(--ink-muted)' }}>
                Queries PubMed (NCBI E-utilities) for published adverse drug reaction literature case reports.
              </p>
              <div className="field-grid">
                <label>
                  <span className="kpi-label">PubMed Search Query</span>
                  <input className="input" value={pubmedQuery} onChange={(e) => setPubmedQuery(e.target.value)} required autoFocus />
                </label>
                <label>
                  <span className="kpi-label">Article Count (1–5)</span>
                  <input className="input" type="number" min={1} max={5} value={limit} onChange={(e) => setLimit(Number(e.target.value))} />
                </label>
              </div>
            </>
          )}

          {mode === 'trials' && (
            <>
              <p style={{ margin: 0, fontSize: '0.84rem', color: 'var(--ink-muted)' }}>
                Queries ClinicalTrials.gov API v2 for completed studies with reported serious/other adverse events.
              </p>
              <div className="field-grid">
                <label>
                  <span className="kpi-label">Drug / Intervention Query</span>
                  <input className="input" value={trialQuery} onChange={(e) => setTrialQuery(e.target.value)} required autoFocus />
                </label>
                <label>
                  <span className="kpi-label">Study Count (1–5)</span>
                  <input className="input" type="number" min={1} max={5} value={limit} onChange={(e) => setLimit(Number(e.target.value))} />
                </label>
              </div>
            </>
          )}

          {mode === 'text' && (
            <>
              <label>
                <span className="kpi-label">Source Channel</span>
                <select className="input" value={sourceType} onChange={(e) => setSourceType(e.target.value)}>
                  <option value="Spontaneous (HCP)">Spontaneous (HCP)</option>
                  <option value="Spontaneous (Patient/Consumer)">Spontaneous (Patient/Consumer)</option>
                  <option value="Literature">Literature</option>
                  <option value="Clinical Trial">Clinical Trial</option>
                </select>
              </label>
              <label>
                <span className="kpi-label">Adverse Event Report Text (Public / Synthetic)</span>
                <textarea rows={5} value={rawText} onChange={(e) => setRawText(e.target.value)} required autoFocus />
              </label>
            </>
          )}

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem' }}>
            <button type="button" className="btn" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? 'Ingesting & Running Agents...' : 'Fetch & Run Pipeline'}
            </button>
          </div>
        </form>
      </div>
    </dialog>
  );
};
