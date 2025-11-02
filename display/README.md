# AusDevs Conversation Analysis - React + Flask

Modern web application for visualizing Discord conversation analysis with an interactive scatter plot powered by Apache ECharts.

## Architecture

- **Backend**: Flask API with SQLite database
- **Frontend**: React + TypeScript with Vite
- **Visualization**: Apache ECharts with WebGL rendering
- **UI Components**: Ant Design dark theme

## Setup

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python backend.py
```

The backend will start on `http://localhost:5000` and serve the following endpoints:

- `GET /api/filters` - Get all filter options
- `GET /api/chunks?vector_type=X&method=Y&...` - Get clustered chunks with filters
- `GET /api/chunk/<chunk_id>` - Get full conversation messages

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will start on `http://localhost:5173`

## Features

### Filters
- **Vector Type**: Choose between topic, sentiment, technical_topic, or combined embeddings
- **Reduction Method**: Select from UMAP, PCA, or PaCMAP dimensionality reduction
- **Channel**: Filter by Discord channel
- **Author**: Filter conversations by specific authors
- **Min Messages**: Show only chunks with at least N messages
- **Search**: Search in topic, technical topic, and sentiment descriptions

### Visualization
- Interactive scatter plot with 400k+ points (GPU-accelerated with WebGL)
- Color-coded clusters with automatic splitting to keep clusters ≤250 points
- Hover tooltips showing chunk metadata
- Click to view full conversation

### Conversation Display
- Full message content with timestamps
- Author names and message text
- Channel and time range information
- Automatically scrollable for large conversations

## Technology Stack

### Backend
- Flask - Web framework
- Flask-CORS - Cross-origin requests
- SQLite - Database
- NumPy/Pandas - Data processing
- scikit-learn - DBSCAN clustering, KMeans splitting, PCA

### Frontend
- React 18 - UI library
- TypeScript - Type safety
- Vite - Build tool
- Apache ECharts - Visualization
- Ant Design - UI components
- Axios - HTTP client

## Data Pipeline

1. User adjusts filters in React frontend
2. Frontend makes API request to Flask backend
3. Backend queries SQLite database for matching chunks
4. Backend applies DBSCAN clustering (eps=0.5, min_samples=2)
5. Backend splits large clusters (>250 points) using KMeans
6. Backend returns JSON with clustered points and metadata
7. Frontend renders scatter plot with ECharts
8. On chunk click, frontend fetches full messages from API
9. Frontend displays conversation with all message details

## Database Schema

### chunks table
```sql
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    channel TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    vector_type TEXT NOT NULL,      -- topic, technical_topic, sentiment, combined
    method TEXT NOT NULL,            -- umap, pca, pacmap
    x REAL NOT NULL,                -- 2D coordinate
    y REAL NOT NULL,                -- 2D coordinate
    cluster INTEGER NOT NULL,
    authors TEXT NOT NULL,          -- JSON array
    msg_count INTEGER NOT NULL,
    chunk_start TEXT NOT NULL,
    chunk_end TEXT NOT NULL,
    topic_short TEXT,
    technical_topic_short TEXT,
    sentiment_short TEXT
)
```

### messages table
```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    chunk_id INTEGER NOT NULL,
    author TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
)
```

## Clustering Algorithm

### Primary: DBSCAN
- `eps=0.5` - Maximum distance between points
- `min_samples=2` - Minimum points to form a cluster
- Returns cluster labels including -1 for noise points

### Secondary: KMeans for Large Cluster Splitting
- Max cluster size: **250 points**
- Iterative splitting (max 20 iterations)
- For clusters > 250 points:
  - Calculate sub-clusters: `n_clusters = max(2, (size + 250 - 1) // 250)`
  - Apply KMeans with `n_clusters`, `n_init=3`, `random_state=42`
  - Assign new labels and repeat until all clusters ≤ 250

## Styling

### Color Scheme
- Background: `#111111` (dark)
- Text: `#cccccc` (light gray)
- Grids: `#333333` (dark gray)
- Cluster colors: Category10 palette (10 colors cycling)
- Primary accent: `#1f77b4` (blue)

### Responsive Design
- Filters panel: 25% width on desktop, 100% on mobile
- Chart: Full width, 600px height
- Conversation panel: Full width, scrollable

## Performance Notes

- Backend clusters data server-side for consistent results
- Frontend uses ECharts WebGL rendering for 400k+ points
- Filter changes debounced by 500ms to avoid excessive API calls
- Message display cached until new chunk is clicked

## Development

### Adding a New Filter
1. Update backend `@app.route("/api/filters")` to include new option
2. Update TypeScript `FilterState` interface to include new field
3. Add Ant Design component in `FilterPanel.tsx`
4. Update `GET /api/chunks` query logic in `backend.py`

### Customizing Colors
- Edit cluster colors in `backend.py`: `colors` variable in `get_chunks()`
- Edit theme in `index.css`: CSS variables for background, text, accent colors
- Edit ECharts theme in `ScatterChart.tsx`: tooltip styling, grid colors, etc.

## Troubleshooting

### Backend Connection Error
- Ensure Flask backend is running on `http://localhost:5000`
- Check CORS is enabled: `CORS(app)` in `backend.py`
- Verify API endpoint with `curl http://localhost:5000/health`

### Slow Clustering
- Large datasets (400k+ chunks) take time to cluster server-side
- Consider pre-computing clusters and caching results
- Increase Flask timeout in `backend.py` if needed

### Chart Not Rendering
- Open browser console for ECharts errors
- Check if `chunks` array is empty in frontend state
- Verify API returns valid chunk data with x, y, color, opacity fields
