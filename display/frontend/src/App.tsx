import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { Alert, Button } from 'antd';
import { MenuOutlined, CloseOutlined } from '@ant-design/icons';
import { FilterPanel } from './components/FilterPanel';
import { ScatterChart } from './components/ScatterChart';
import { ConversationDisplay } from './components/ConversationDisplay';
import type { FilterOptions, FilterState, Chunk, ConversationDetail } from './types';
import './App.css';

const API_BASE = 'http://localhost:5000/api';

function App() {
  // Initialize sidebar based on screen width
  const [sidebarOpen, setSidebarOpen] = useState(() => {
    if (typeof window === 'undefined') return true;
    return window.innerWidth > 768;
  });

  const [filterOptions, setFilterOptions] = useState<FilterOptions | null>(null);
  const [filters, setFilters] = useState<FilterState>({
    vector_type: 'combined',
    method: 'umap',
    channel: 'All Channels',
    author: 'All Authors',
    cluster: 'All Clusters',
    min_messages: 5,
    search: '',
  });

  // Ensure UMAP is default method
  const [methodSet, setMethodSet] = useState(false);

  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [conversation, setConversation] = useState<ConversationDetail | null>(null);

  const [loading, setLoading] = useState(false);
  const [conversationLoading, setConversationLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Handle sidebar visibility on window resize
  useEffect(() => {
    const handleResize = () => {
      setSidebarOpen(window.innerWidth > 768);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Fetch filter options on mount
  useEffect(() => {
    const fetchFilters = async () => {
      try {
        const response = await axios.get<FilterOptions>(`${API_BASE}/filters`);
        setFilterOptions(response.data);
        // Set defaults based on first options
        if (response.data.vector_types.length > 0) {
          setFilters(prev => ({
            ...prev,
            vector_type: response.data.vector_types[0],
          }));
        }
        // Prefer UMAP as default method
        if (response.data.methods.length > 0) {
          const defaultMethod = response.data.methods.includes('umap')
            ? 'umap'
            : response.data.methods[0];
          setFilters(prev => ({
            ...prev,
            method: defaultMethod,
          }));
          setMethodSet(true);
        }
      } catch (err) {
        setError(`Failed to load filter options: ${err}`);
      }
    };

    fetchFilters();
  }, []);

  // Fetch chunks when filters change
  useEffect(() => {
    const fetchChunks = async () => {
      if (!filterOptions) return;

      setLoading(true);
      setChunks([]);  // Clear old data immediately
      setError(null);
      setConversation(null);

      try {
        // Only request vector_type and method from server
        const params = new URLSearchParams({
          vector_type: filters.vector_type,
          method: filters.method,
        });

        const response = await axios.get<{ chunks: Chunk[] }>(
          `${API_BASE}/chunks?${params}`
        );

        // Client-side filtering
        let filteredChunks = response.data.chunks;

        // Filter by channel
        if (filters.channel !== 'All Channels') {
          filteredChunks = filteredChunks.filter(
            chunk => chunk.channel === filters.channel
          );
        }

        // Filter by author
        if (filters.author !== 'All Authors') {
          filteredChunks = filteredChunks.filter(chunk =>
            chunk.authors.split(', ').includes(filters.author)
          );
        }

        // Filter by cluster
        if (filters.cluster !== 'All Clusters') {
          // Extract cluster number from "Cluster 140 - 501 chunks" format
          const clusterNum = parseInt(filters.cluster.match(/\d+/)?.[0] || '0', 10);
          filteredChunks = filteredChunks.filter(chunk =>
            chunk.cluster === clusterNum
          );
        }

        // Filter by min_messages
        filteredChunks = filteredChunks.filter(
          chunk => chunk.messages >= filters.min_messages
        );

        // Filter by search text
        if (filters.search) {
          const searchLower = filters.search.toLowerCase();
          filteredChunks = filteredChunks.filter(chunk =>
            chunk.topic.toLowerCase().includes(searchLower) ||
            chunk.technical_topic.toLowerCase().includes(searchLower) ||
            chunk.sentiment.toLowerCase().includes(searchLower)
          );
        }

        setChunks(filteredChunks);
      } catch (err) {
        setError(`Failed to load chunks: ${err}`);
      } finally {
        setLoading(false);
      }
    };

    const timer = setTimeout(() => {
      fetchChunks();
    }, 500);

    return () => clearTimeout(timer);
  }, [filters.vector_type, filters.method, filters.channel, filters.author, filters.cluster, filters.min_messages, filters.search, filterOptions]);

  // Fetch conversation when a chunk is clicked
  const handleChunkClick = useCallback(async (chunk: Chunk) => {
    setConversationLoading(true);
    setError(null);

    try {
      const response = await axios.get<ConversationDetail>(
        `${API_BASE}/chunk/${chunk.id}`
      );
      setConversation(response.data);
    } catch (err) {
      setError(`Failed to load conversation: ${err}`);
    } finally {
      setConversationLoading(false);
    }
  }, []);

  return (
    <div className="app">
      <div className="app-header">
        <div className="header-content">
          <div className="header-logo">
            <img src="/favicon.svg" alt="AusDevs" className="logo-icon" />
          </div>
          <Button
            type="text"
            icon={sidebarOpen ? <CloseOutlined /> : <MenuOutlined />}
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="sidebar-toggle"
            aria-label={sidebarOpen ? 'Close filters' : 'Open filters'}
          />
          <div className="header-text">
            <h1>AusDevs 2.0.0 Conversation Analysis</h1>
            <p>Explore 40,000+ conversation chunks from the AusDevs 2.0.0 Discord server. Click a point to see the conversation.</p>
            <a
              href="https://zyros.notion.site/How-I-made-AusDevs-2-0-0-Conversation-Analysis-29fd636b94f0802fad48cf07d5c8f4c3"
              target="_blank"
              rel="noopener noreferrer"
              className="header-link"
            >
              How this was made
            </a>
          </div>
        </div>
      </div>

      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: '20px' }}
        />
      )}

      <div className={`app-content ${sidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
        <div className={`filter-sidebar ${sidebarOpen ? 'visible' : 'hidden'}`}>
          <FilterPanel
            options={filterOptions}
            filters={filters}
            loading={loading}
            onFiltersChange={setFilters}
          />
        </div>

        {sidebarOpen && <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />}

        <div className="main-panel">
          <div className="chart-panel">
            <ScatterChart
              chunks={chunks}
              filters={filters}
              onChunkClick={handleChunkClick}
              loading={loading}
            />
          </div>

          <div className="conversation-panel">
            <ConversationDisplay
              conversation={conversation}
              loading={conversationLoading}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
