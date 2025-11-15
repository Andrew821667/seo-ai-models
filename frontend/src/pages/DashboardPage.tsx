/**
 * Dashboard Page
 *
 * Main dashboard showing active analyses and system stats
 */

import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { useWebSocket } from '../hooks/useWebSocket';
import { analysisAPI } from '../lib/api';
import { useAuthStore } from '../stores/authStore';
import { Activity, TrendingUp, Users, Clock } from 'lucide-react';

interface Analysis {
  analysis_id: string;
  status: string;
  progress: number;
  current_step: string;
  url: string;
  started_at: string;
}

export default function DashboardPage() {
  const navigate = useNavigate();
  const { user } = useAuthStore();
  const [activeAnalyses, setActiveAnalyses] = useState<Analysis[]>([]);

  // Fetch active analyses
  const { data: analyses } = useQuery({
    queryKey: ['active-analyses'],
    queryFn: analysisAPI.getActiveAnalyses,
    refetchInterval: 5000, // Refetch every 5 seconds
  });

  // WebSocket for real-time updates
  const { lastMessage } = useWebSocket({
    room: 'global',
    onMessage: (message) => {
      if (message.type === 'analysis_update') {
        // Update active analyses in real-time
        setActiveAnalyses((prev) => {
          const index = prev.findIndex((a) => a.analysis_id === message.data.id);
          if (index >= 0) {
            const updated = [...prev];
            updated[index] = message.data;
            return updated;
          } else {
            return [...prev, message.data];
          }
        });
      }
    },
  });

  useEffect(() => {
    if (analyses) {
      setActiveAnalyses(analyses);
    }
  }, [analyses]);

  const stats = [
    {
      label: 'Active Analyses',
      value: activeAnalyses.filter((a) => a.status === 'started').length,
      icon: Activity,
      color: 'blue',
    },
    {
      label: 'Completed Today',
      value: activeAnalyses.filter((a) => a.status === 'completed').length,
      icon: TrendingUp,
      color: 'green',
    },
    {
      label: 'Your Role',
      value: user?.role || 'N/A',
      icon: Users,
      color: 'purple',
    },
    {
      label: 'Queue',
      value: activeAnalyses.filter((a) => a.progress === 0).length,
      icon: Clock,
      color: 'orange',
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-1">
            Welcome back, {user?.full_name || user?.username}
          </p>
        </div>

        <button
          onClick={() => navigate('/analysis')}
          className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-semibold transition"
        >
          + New Analysis
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <div
              key={stat.label}
              className="bg-white rounded-xl shadow p-6 border-l-4"
              style={{ borderColor: `var(--${stat.color}-500)` }}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 font-medium">{stat.label}</p>
                  <p className="text-3xl font-bold text-gray-900 mt-2">{stat.value}</p>
                </div>
                <div className={`p-3 bg-${stat.color}-100 rounded-lg`}>
                  <Icon className={`w-8 h-8 text-${stat.color}-600`} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Active Analyses */}
      <div className="bg-white rounded-xl shadow">
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">Active Analyses</h2>
        </div>

        <div className="divide-y divide-gray-200">
          {activeAnalyses.length === 0 ? (
            <div className="p-12 text-center text-gray-500">
              <Activity className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              <p>No active analyses</p>
              <p className="text-sm mt-2">Start a new analysis to see real-time progress here</p>
            </div>
          ) : (
            activeAnalyses.map((analysis) => (
              <div
                key={analysis.analysis_id}
                className="p-6 hover:bg-gray-50 cursor-pointer transition"
                onClick={() => navigate(`/analysis/${analysis.analysis_id}`)}
              >
                <div className="flex items-center justify-between mb-4">
                  <div className="flex-1">
                    <h3 className="font-semibold text-gray-900">{analysis.url}</h3>
                    <p className="text-sm text-gray-600 mt-1">{analysis.current_step}</p>
                  </div>
                  <span
                    className={`px-3 py-1 rounded-full text-sm font-medium ${
                      analysis.status === 'completed'
                        ? 'bg-green-100 text-green-800'
                        : analysis.status === 'failed'
                        ? 'bg-red-100 text-red-800'
                        : 'bg-blue-100 text-blue-800'
                    }`}
                  >
                    {analysis.status}
                  </span>
                </div>

                {/* Progress Bar */}
                <div className="relative h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="absolute top-0 left-0 h-full bg-blue-600 transition-all duration-500"
                    style={{ width: `${analysis.progress}%` }}
                  />
                </div>
                <div className="flex justify-between items-center mt-2">
                  <span className="text-sm text-gray-600">{analysis.progress}%</span>
                  <span className="text-xs text-gray-500">
                    Started {new Date(analysis.started_at).toLocaleTimeString()}
                  </span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
