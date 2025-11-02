.PHONY: help install compute launch stop backend frontend clean

help:
	@echo "AusDevs Conversation Analysis - Makefile Commands"
	@echo ""
	@echo "  make install  - Install all dependencies (backend + frontend)"
	@echo "  make compute  - Pre-compute cache data (run once or periodically)"
	@echo "  make launch   - Launch both backend and frontend servers"
	@echo "  make backend  - Start Flask backend only (pure key-value server)"
	@echo "  make frontend - Start React frontend only (requires backend running)"
	@echo "  make stop     - Stop all running servers"
	@echo "  make clean    - Clean node_modules and Python cache"
	@echo "  make help     - Show this help message"

install:
	@echo "Installing backend dependencies..."
	cd backend && pip install -r requirements.txt
	@echo "Installing frontend dependencies..."
	cd frontend && npm install
	@echo "✓ All dependencies installed"

compute:
	@echo "Pre-computing cache data..."
	cd backend && python compute_data.py

launch: compute
	@echo ""
	@echo "Starting AusDevs Conversation Analysis..."
	@echo ""
	@echo "Starting Flask backend on http://localhost:5000..."
	cd backend && python backend.py > /tmp/backend.log 2>&1 &
	@sleep 2
	@echo "Starting React frontend on http://localhost:5173..."
	cd frontend && npm run dev > /tmp/frontend.log 2>&1 &
	@sleep 3
	@echo ""
	@echo "✓ Both servers started!"
	@echo ""
	@echo "Frontend: http://localhost:5173"
	@echo "Backend:  http://localhost:5000"
	@echo ""
	@echo "Logs:"
	@echo "  Backend:  tail -f /tmp/backend.log"
	@echo "  Frontend: tail -f /tmp/frontend.log"
	@echo ""
	@echo "To stop servers: make stop"

backend:
	@echo "Starting Flask backend on http://localhost:5000 (pure key-value server)..."
	cd backend && python backend.py

frontend:
	@echo "Starting React frontend on http://localhost:5173..."
	@echo "Make sure backend is running on http://localhost:5000"
	cd frontend && npm run dev

stop:
	@echo "Stopping all servers..."
	@pkill -f "python backend.py" || true
	@pkill -f "npm run dev" || true
	@pkill -f "vite" || true
	@echo "✓ All servers stopped"

clean:
	@echo "Cleaning up..."
	rm -rf frontend/node_modules
	rm -rf frontend/.vite
	rm -rf frontend/dist
	find backend -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find backend -type f -name "*.pyc" -delete
	@echo "✓ Cleanup complete"
