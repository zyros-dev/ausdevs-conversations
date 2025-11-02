.PHONY: help server compute install clean

help:
	@echo "AusDevs Backend - Available commands:"
	@echo ""
	@echo "  make compute  - Pre-compute all cache data (vector_type + method combinations)"
	@echo "  make server   - Start the Flask backend server (pure key-value lookup, no computation)"
	@echo "  make install  - Install Python dependencies"
	@echo "  make clean    - Clear all cache entries from database"
	@echo ""

install:
	pip install -r requirements.txt

compute:
	python compute_data.py

server:
	python backend.py

clean:
	@echo "Clearing cache from database..."
	python -c "import sqlite3; conn = sqlite3.connect('conversation_data.db'); conn.cursor().execute('DELETE FROM cache'); conn.commit(); conn.close(); print('✓ Cache cleared')"
