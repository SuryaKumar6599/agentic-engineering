export type EnterpriseRole =
  | 'CASE_PROCESSOR'
  | 'QC_REVIEWER'
  | 'MEDICAL_REVIEWER'
  | 'PROJECT_LEAD'
  | 'ADMIN';

export interface EnterpriseUser {
  username: string;
  full_name: string;
  email: string;
  role: EnterpriseRole;
  department: string;
  permissions: string[];
}

const TOKEN_KEY = 'pv_auth_token';
const USER_KEY = 'pv_auth_user';

export function getAuthToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function getStoredUser(): EnterpriseUser | null {
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as EnterpriseUser;
  } catch {
    return null;
  }
}

export function setAuthSession(token: string, user: EnterpriseUser): void {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function clearAuthSession(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export function hasPerm(user: EnterpriseUser | null, perm: string): boolean {
  return Boolean(user && user.permissions.includes(perm));
}

export async function apiFetch(input: string, init: RequestInit = {}): Promise<Response> {
  const token = getAuthToken();
  const headers = new Headers(init.headers || {});
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  return fetch(input, { ...init, headers });
}
