/**
 * Analysis Page
 *
 * Create new analysis or view analysis details with real-time progress
 */

import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { analysisAPI } from '../lib/api';
import { useWebSocket } from '../hooks/useWebSocket';
import { ArrowLeft, Play, CheckCircle, XCircle, AlertCircle } from 'lucide-react';

export default function AnalysisPage() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    url: '',
    content: '',
    keywords: '',
    auto_fix: true,
    fix_complexity_limit: 'simple',
  });

  const [analysis, setAnalysis] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // WebSocket for specific analysis
  const { lastMessage, isConnected } = useWebSocket({
    analysisId: id,
    onMessage: (message) => {
      if (message.type === 'analysis_update' || message.type === 'analysis_state') {
        setAnalysis(message.data);
      }
    },
  });

  // Fetch existing analysis
  useEffect(() => {
    if (id) {
      analysisAPI.getStatus(id)
        .then(setAnalysis)
        .catch((err) => setError(err.message));
    }
  }, [id]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const keywords = formData.keywords.split(',').map((k) => k.trim());

      const result = await analysisAPI.startAnalysis({
        url: formData.url,
        content: formData.content,
        keywords,
        auto_fix: formData.auto_fix,
        fix_complexity_limit: formData.fix_complexity_limit,
      });

      // Navigate to analysis view
      navigate(`/analysis/${result.analysis_id}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start analysis');
    } finally {
      setLoading(false);
    }
  };

  // If viewing existing analysis
  if (id && analysis) {
    return (
      <div className="max-w-4xl mx-auto space-y-6">
        <button
          onClick={() => navigate('/dashboard')}
          className="flex items-center text-gray-600 hover:text-gray-900"
        >
          <ArrowLeft className="w-5 h-5 mr-2" />
          Back to Dashboard
        </button>

        <div className="bg-white rounded-xl shadow p-8">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-2xl font-bold text-gray-900">Analysis Details</h1>
            <div className="flex items-center">
              <div
                className={`w-3 h-3 rounded-full mr-2 ${
                  isConnected ? 'bg-green-500' : 'bg-red-500'
                }`}
              />
              <span className="text-sm text-gray-600">
                {isConnected ? 'Live' : 'Disconnected'}
              </span>
            </div>
          </div>

          {/* URL */}
          <div className="mb-6">
            <label className="text-sm font-medium text-gray-700">URL</label>
            <p className="text-gray-900 mt-1">{analysis.url}</p>
          </div>

          {/* Status */}
          <div className="mb-6">
            <label className="text-sm font-medium text-gray-700 mb-2 block">Status</label>
            <div className="flex items-center">
              {analysis.status === 'completed' && (
                <CheckCircle className="w-6 h-6 text-green-600 mr-2" />
              )}
              {analysis.status === 'failed' && (
                <XCircle className="w-6 h-6 text-red-600 mr-2" />
              )}
              {analysis.status === 'started' && (
                <AlertCircle className="w-6 h-6 text-blue-600 mr-2 animate-pulse" />
              )}
              <span className="text-lg font-semibold capitalize">{analysis.status}</span>
            </div>
          </div>

          {/* Progress */}
          <div className="mb-6">
            <div className="flex justify-between items-center mb-2">
              <label className="text-sm font-medium text-gray-700">Progress</label>
              <span className="text-sm font-semibold text-gray-900">{analysis.progress}%</span>
            </div>
            <div className="relative h-4 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="absolute top-0 left-0 h-full bg-blue-600 transition-all duration-500"
                style={{ width: `${analysis.progress}%` }}
              />
            </div>
            <p className="text-sm text-gray-600 mt-2">{analysis.current_step}</p>
          </div>

          {/* Steps History */}
          {analysis.steps && analysis.steps.length > 0 && (
            <div className="mb-6">
              <label className="text-sm font-medium text-gray-700 mb-3 block">Progress Steps</label>
              <div className="space-y-3">
                {analysis.steps.map((step: any, index: number) => (
                  <div key={index} className="flex items-center text-sm">
                    <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center mr-3">
                      <span className="text-blue-600 font-semibold">{index + 1}</span>
                    </div>
                    <div className="flex-1">
                      <p className="text-gray-900">{step.step}</p>
                      <p className="text-xs text-gray-500">
                        {new Date(step.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                    <span className="text-gray-600">{step.progress}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Results */}
          {analysis.results && (
            <div className="mb-6">
              <label className="text-sm font-medium text-gray-700 mb-3 block">Results</label>
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="text-sm text-gray-600 mb-2">
                  Overall Score: <span className="font-bold text-lg">{analysis.results.overall_score}/100</span>
                </p>
                <p className="text-sm text-gray-600 mb-2">
                  Issues Detected: <span className="font-semibold">{analysis.results.issues_detected?.length || 0}</span>
                </p>
                <p className="text-sm text-gray-600">
                  Fixes Applied: <span className="font-semibold text-green-600">{analysis.results.fixes_applied?.length || 0}</span>
                </p>
              </div>
            </div>
          )}

          {/* Error */}
          {analysis.error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
              <p className="font-semibold">Error:</p>
              <p className="mt-1">{analysis.error}</p>
            </div>
          )}
        </div>
      </div>
    );
  }

  // New analysis form
  return (
    <div className="max-w-2xl mx-auto">
      <button
        onClick={() => navigate('/dashboard')}
        className="flex items-center text-gray-600 hover:text-gray-900 mb-6"
      >
        <ArrowLeft className="w-5 h-5 mr-2" />
        Back to Dashboard
      </button>

      <div className="bg-white rounded-xl shadow p-8">
        <h1 className="text-2xl font-bold text-gray-900 mb-6">New SEO Analysis</h1>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-6">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              URL to Analyze *
            </label>
            <input
              type="url"
              value={formData.url}
              onChange={(e) => setFormData({ ...formData, url: e.target.value })}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="https://example.com"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Content *
            </label>
            <textarea
              value={formData.content}
              onChange={(e) => setFormData({ ...formData, content: e.target.value })}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={6}
              placeholder="Paste your content here..."
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Keywords (comma-separated) *
            </label>
            <input
              type="text"
              value={formData.keywords}
              onChange={(e) => setFormData({ ...formData, keywords: e.target.value })}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="seo, optimization, keywords"
              required
            />
          </div>

          <div>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={formData.auto_fix}
                onChange={(e) => setFormData({ ...formData, auto_fix: e.target.checked })}
                className="w-5 h-5 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
              />
              <span className="ml-3 text-sm font-medium text-gray-700">
                Enable Auto-Fix
              </span>
            </label>
          </div>

          {formData.auto_fix && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Fix Complexity Limit
              </label>
              <select
                value={formData.fix_complexity_limit}
                onChange={(e) => setFormData({ ...formData, fix_complexity_limit: e.target.value })}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="trivial">Trivial (very safe)</option>
                <option value="simple">Simple (safe)</option>
                <option value="moderate">Moderate (review recommended)</option>
                <option value="complex">Complex (requires approval)</option>
              </select>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-semibold py-3 px-4 rounded-lg transition duration-200 flex items-center justify-center"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2" />
                Starting Analysis...
              </>
            ) : (
              <>
                <Play className="w-5 h-5 mr-2" />
                Start Analysis
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
