/**
 * Authentication Store
 *
 * Manages user authentication state with Zustand
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface User {
  id: string;
  username: string;
  email: string;
  role: 'admin' | 'analyst' | 'user' | 'observer';
  full_name?: string;
  is_active: boolean;
  created_at: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  setUser: (user: User, token: string) => void;
  logout: () => void;
  hasPermission: (permission: string) => boolean;
}

const ROLE_PERMISSIONS = {
  admin: ['*'], // All permissions
  analyst: ['run_analysis', 'view_analysis', 'delete_analysis', 'execute_autofix', 'approve_autofix', 'view_system_stats'],
  user: ['run_analysis', 'view_analysis', 'execute_autofix'],
  observer: ['view_analysis', 'view_system_stats'],
};

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,

      setUser: (user, token) => {
        set({
          user,
          token,
          isAuthenticated: true,
        });
      },

      logout: () => {
        set({
          user: null,
          token: null,
          isAuthenticated: false,
        });
      },

      hasPermission: (permission) => {
        const { user } = get();
        if (!user) return false;

        const permissions = ROLE_PERMISSIONS[user.role] || [];
        return permissions.includes('*') || permissions.includes(permission);
      },
    }),
    {
      name: 'auth-storage',
    }
  )
);
