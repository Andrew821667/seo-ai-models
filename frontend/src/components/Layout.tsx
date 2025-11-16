/**
 * Main Layout Component
 *
 * Provides navigation and layout structure
 */

import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import { LayoutDashboard, Activity, Shield, LogOut, Menu, X } from 'lucide-react';
import { useState } from 'react';

export default function Layout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuthStore();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const navigation = [
    {
      name: 'Dashboard',
      href: '/dashboard',
      icon: LayoutDashboard,
      show: true,
    },
    {
      name: 'New Analysis',
      href: '/analysis',
      icon: Activity,
      show: true,
    },
    {
      name: 'Admin Panel',
      href: '/admin',
      icon: Shield,
      show: user?.role === 'admin',
    },
  ];

  const isActive = (href: string) => location.pathname === href;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top Navigation */}
      <nav className="bg-white border-b border-gray-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-bold text-gray-900">SEO AI Models</h1>

              {/* Desktop Navigation */}
              <div className="hidden md:flex ml-10 space-x-4">
                {navigation
                  .filter((item) => item.show)
                  .map((item) => {
                    const Icon = item.icon;
                    return (
                      <button
                        key={item.name}
                        onClick={() => navigate(item.href)}
                        className={`flex items-center px-3 py-2 rounded-lg text-sm font-medium transition ${
                          isActive(item.href)
                            ? 'bg-blue-50 text-blue-700'
                            : 'text-gray-700 hover:bg-gray-100'
                        }`}
                      >
                        <Icon className="w-5 h-5 mr-2" />
                        {item.name}
                      </button>
                    );
                  })}
              </div>
            </div>

            <div className="flex items-center">
              {/* User Info */}
              <div className="hidden md:flex items-center mr-4">
                <div className="text-right mr-3">
                  <p className="text-sm font-semibold text-gray-900">{user?.username}</p>
                  <p className="text-xs text-gray-600 capitalize">{user?.role}</p>
                </div>
                <button
                  onClick={handleLogout}
                  className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition"
                  title="Logout"
                >
                  <LogOut className="w-5 h-5" />
                </button>
              </div>

              {/* Mobile Menu Button */}
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="md:hidden p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition"
              >
                {mobileMenuOpen ? (
                  <X className="w-6 h-6" />
                ) : (
                  <Menu className="w-6 h-6" />
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-gray-200 bg-white">
            <div className="px-4 py-4 space-y-2">
              {navigation
                .filter((item) => item.show)
                .map((item) => {
                  const Icon = item.icon;
                  return (
                    <button
                      key={item.name}
                      onClick={() => {
                        navigate(item.href);
                        setMobileMenuOpen(false);
                      }}
                      className={`w-full flex items-center px-3 py-2 rounded-lg text-sm font-medium transition ${
                        isActive(item.href)
                          ? 'bg-blue-50 text-blue-700'
                          : 'text-gray-700 hover:bg-gray-100'
                      }`}
                    >
                      <Icon className="w-5 h-5 mr-2" />
                      {item.name}
                    </button>
                  );
                })}

              <div className="pt-4 border-t border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-semibold text-gray-900">{user?.username}</p>
                    <p className="text-xs text-gray-600 capitalize">{user?.role}</p>
                  </div>
                  <button
                    onClick={handleLogout}
                    className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition"
                  >
                    <LogOut className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Outlet />
      </main>
    </div>
  );
}
