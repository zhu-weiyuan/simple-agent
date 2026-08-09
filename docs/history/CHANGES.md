# Changelog

## 2026-06-27 — UI Overhaul & Bug Fixes

### 🐛 Bug Fixes (app.py)
- **Fixed duplicate FastAPI initialization** — Two `app = FastAPI(...)` declarations caused the second to overwrite the first, losing `json_response_class=UTF8JSONResponse`. Removed the duplicate.
- **Fixed missing `Response` import** — `Response` was used in the `/api/chat` endpoint but never imported from FastAPI. Added to import statement.

### 🎨 Frontend Redesign (web/index.html)
- **Complete UI redesign** with modern dark theme (indigo/purple palette)
- **Glassmorphism header** with logo, status badge, and action buttons
- **Welcome screen** with animated icon and feature badges
- **Improved message bubbles** — gradient user messages, accent-bordered assistant messages
- **Message actions** — copy button appears on hover for assistant messages
- **Intent analysis panel** — slide-in panel with confidence bars
- **Quick actions bar** — rounded pill buttons for common tasks
- **Character counter** — live count in input field
- **Smooth animations** — message entrance, typing indicator, panel transitions
- **Light/dark theme toggle** with localStorage persistence
- **Responsive design** — mobile-friendly layout
- **Custom scrollbar** — subtle, matches theme

### 🧹 Cleanup
- **Removed `static/` directory** — Conflicting frontend (light theme) that was never served. The app serves from `web/`.
- **Updated README.md** — Accurate feature list, clean architecture diagram, correct API documentation
