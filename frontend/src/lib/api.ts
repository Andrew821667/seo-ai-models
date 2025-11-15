/**
 * API Client
 *
 * Axios-based client for backend communication
 */

import axios from 'axios';
import { useAuthStore } from '../stores/authStore';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v2';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
apiClient.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle auth errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout();
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authAPI = {
  login: async (username: string, password: string) => {
    const response = await apiClient.post('/auth/login', { username, password });
    return response.data;
  },

  logout: async () => {
    await apiClient.post('/auth/logout');
  },

  getCurrentUser: async () => {
    const response = await apiClient.get('/auth/me');
    return response.data;
  },

  getPermissions: async () => {
    const response = await apiClient.get('/auth/me/permissions');
    return response.data;
  },
};

// Analysis API
export const analysisAPI = {
  startAnalysis: async (data: {
    url: string;
    content: string;
    keywords: string[];
    auto_fix?: boolean;
    fix_complexity_limit?: string;
  }) => {
    const response = await apiClient.post('/enhanced-analysis/analyze', data);
    return response.data;
  },

  getStatus: async (analysisId: string) => {
    const response = await apiClient.get(`/enhanced-analysis/status/${analysisId}`);
    return response.data;
  },

  getActiveAnalyses: async () => {
    const response = await apiClient.get('/enhanced-analysis/active');
    return response.data;
  },

  approveAutoFix: async (actionId: string, approved: boolean, reason?: string) => {
    const response = await apiClient.post('/enhanced-analysis/autofix/approve', {
      action_id: actionId,
      approved,
      reason,
    });
    return response.data;
  },

  getPendingAutoFixes: async () => {
    const response = await apiClient.get('/enhanced-analysis/autofix/pending');
    return response.data;
  },
};

// Admin API
export const adminAPI = {
  listUsers: async () => {
    const response = await apiClient.get('/auth/users');
    return response.data;
  },

  createUser: async (userData: {
    username: string;
    email: string;
    password: string;
    role: string;
    full_name?: string;
  }) => {
    const response = await apiClient.post('/auth/users', userData);
    return response.data;
  },

  updateUser: async (userId: string, userData: Partial<{
    username: string;
    email: string;
    role: string;
    full_name: string;
    is_active: boolean;
  }>) => {
    const response = await apiClient.patch(`/auth/users/${userId}`, userData);
    return response.data;
  },

  deleteUser: async (userId: string) => {
    await apiClient.delete(`/auth/users/${userId}`);
  },

  listRoles: async () => {
    const response = await apiClient.get('/auth/roles');
    return response.data;
  },
};
