# SEO AI Models - Frontend

React + TypeScript frontend with real-time WebSocket monitoring.

## Features

- ✅ **Role-based Authentication** (Admin, Analyst, User, Observer)
- ✅ **Real-time Analysis Monitoring** via WebSocket
- ✅ **Dashboard** with live updates
- ✅ **Admin Panel** for user management
- ✅ **Analysis Interface** to create and track SEO analyses
- ✅ **Auto-fix Progress Tracking** in real-time

## Tech Stack

- **React 18** with TypeScript
- **React Router** for navigation
- **Zustand** for state management
- **TanStack Query** for server state
- **Tailwind CSS** for styling
- **WebSocket** for real-time updates
- **Axios** for API calls
- **Vite** for build tool

## Getting Started

### Install dependencies

```bash
npm install
```

### Environment Variables

Create `.env` file:

```env
VITE_API_URL=http://localhost:8000/api/v2
VITE_WS_URL=ws://localhost:8000
```

### Development

```bash
npm run dev
```

App will run on `http://localhost:3000`

### Build for Production

```bash
npm run build
```

## Project Structure

```
src/
├── components/      # Reusable components
│   └── Layout.tsx   # Main layout with navigation
├── pages/           # Page components
│   ├── LoginPage.tsx
│   ├── DashboardPage.tsx
│   ├── AnalysisPage.tsx
│   └── AdminPage.tsx
├── stores/          # Zustand stores
│   └── authStore.ts
├── hooks/           # Custom hooks
│   └── useWebSocket.ts
├── lib/             # Utilities
│   └── api.ts       # API client
├── App.tsx          # Main app component
├── main.tsx         # Entry point
└── index.css        # Global styles
```

## User Roles

### Admin
- Full access to all features
- User management (create, edit, delete users)
- Assign roles
- Approve complex auto-fixes

### Analyst
- Run and view analyses
- Execute and approve auto-fixes
- View system stats

### User
- Run own analyses
- Execute simple auto-fixes
- View own results

### Observer
- Read-only access
- View analyses and stats
- No modification permissions

## WebSocket Events

The frontend connects to WebSocket for real-time updates:

```typescript
// Connect to global room
ws://localhost:8000/ws?token={JWT}&room=global

// Connect to specific analysis
ws://localhost:8000/ws/analysis/{analysisId}?token={JWT}
```

### Message Types

- `connection` - Connection status
- `analysis_update` - Analysis progress update
- `autofix_update` - Auto-fix execution update
- `system_stats` - System statistics

## API Integration

All API calls use JWT authentication:

```typescript
Authorization: Bearer {token}
```

### Endpoints

- `POST /api/v2/auth/login` - Login
- `GET /api/v2/auth/me` - Get current user
- `POST /api/v2/enhanced-analysis/analyze` - Start analysis
- `GET /api/v2/enhanced-analysis/status/{id}` - Get analysis status
- `GET /api/v2/auth/users` - List users (Admin only)
- `POST /api/v2/auth/users` - Create user (Admin only)

## Development Notes

- All components use TypeScript for type safety
- Tailwind CSS for styling with custom configuration
- WebSocket reconnects automatically on disconnect
- Real-time progress updates without polling
- Optimistic UI updates for better UX
