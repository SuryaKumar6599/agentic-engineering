import React, { useEffect, useRef, useState } from 'react';
import { KeyRound, Shield, X } from 'lucide-react';
import { EnterpriseUser, setAuthSession } from '../auth';

interface RoleLoginModalProps {
  open: boolean;
  onClose: () => void;
  onAuthenticated: (user: EnterpriseUser) => void;
  allowCancel?: boolean;
}

export const RoleLoginModal: React.FC<RoleLoginModalProps> = ({
  open,
  onClose,
  onAuthenticated,
  allowCancel = true,
}) => {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [users, setUsers] = useState<EnterpriseUser[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const dlg = dialogRef.current;
    if (!dlg) return;
    if (open && !dlg.open) dlg.showModal();
    else if (!open && dlg.open) dlg.close();
  }, [open]);

  useEffect(() => {
    if (open) {
      fetch('/api/auth/users')
        .then((r) => r.json())
        .then((data) => setUsers(data))
        .catch(() => {});
    }
  }, [open]);

  const loginAs = async (username: string) => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password: 'pv-demo-password' }),
      });
      if (!res.ok) throw new Error('Authentication failed');
      const data = await res.json();
      setAuthSession(data.token, data.user);
      onAuthenticated(data.user);
      onClose();
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <dialog
      ref={dialogRef}
      className="pv-dialog"
      aria-labelledby="rbac-dialog-title"
      onCancel={(e) => {
        if (!allowCancel) e.preventDefault();
        else onClose();
      }}
    >
      <div className="dialog-head">
        <h1 id="rbac-dialog-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Shield size={18} color="var(--accent)" /> Enterprise Role-Based Access (RBAC)
        </h1>
        {allowCancel && (
          <button type="button" className="btn" onClick={onClose} aria-label="Close dialog">
            <X size={16} />
          </button>
        )}
      </div>

      <div className="dialog-body">
        <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--ink-muted)' }}>
          Select an enterprise pharmacovigilance role to enforce GxP separation of duties across Intake, Case Processing, QC, Medical Review, and Project Lead Approval.
        </p>

        {error && <div className="badge badge-critical">{error}</div>}

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
          {users.map((u) => (
            <button
              key={u.username}
              type="button"
              className="card"
              disabled={loading}
              onClick={() => loginAs(u.username)}
              style={{
                textAlign: 'left',
                cursor: 'pointer',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                gap: '0.75rem',
                padding: '0.75rem 1rem',
              }}
            >
              <div>
                <div style={{ fontWeight: 700, fontSize: '0.92rem' }}>
                  {u.full_name} <span className="badge badge-standard" style={{ marginLeft: 6 }}>{u.role}</span>
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--ink-muted)', marginTop: 2 }}>
                  {u.department} • {u.email}
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--ink-muted)', marginTop: 4 }}>
                  Permissions: {u.permissions.join(', ')}
                </div>
              </div>
              <span className="btn btn-primary" style={{ flexShrink: 0 }}>
                <KeyRound size={14} /> Sign In
              </span>
            </button>
          ))}
        </div>
      </div>
    </dialog>
  );
};
