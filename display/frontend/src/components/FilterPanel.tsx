import React from 'react';
import { Select, Slider, Input, Row, Col, Spin } from 'antd';
import type { FilterOptions, FilterState } from '../types';
import './FilterPanel.css';

interface FilterPanelProps {
  options: FilterOptions | null;
  filters: FilterState;
  loading: boolean;
  onFiltersChange: (filters: FilterState) => void;
}

export const FilterPanel: React.FC<FilterPanelProps> = ({
  options,
  filters,
  loading,
  onFiltersChange,
}) => {
  if (!options) {
    return <Spin />;
  }

  const handleVectorTypeChange = (value: string) => {
    onFiltersChange({ ...filters, vector_type: value });
  };

  const handleMethodChange = (value: string) => {
    onFiltersChange({ ...filters, method: value });
  };

  const handleChannelChange = (value: string) => {
    onFiltersChange({ ...filters, channel: value });
  };

  const handleAuthorChange = (value: string) => {
    onFiltersChange({ ...filters, author: value });
  };

  const handleClusterChange = (value: string) => {
    onFiltersChange({ ...filters, cluster: value });
  };

  const handleMinMessagesChange = (value: number | number[]) => {
    const val = Array.isArray(value) ? value[0] : value;
    onFiltersChange({ ...filters, min_messages: val });
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onFiltersChange({ ...filters, search: e.target.value });
  };

  return (
    <div className="filter-panel">
      <h2>Filters</h2>

      <Row gutter={[16, 16]}>
        <Col xs={24}>
          <div className="filter-group">
            <label>Vector Type</label>
            <Select
              value={filters.vector_type}
              onChange={handleVectorTypeChange}
              options={options.vector_types.map(vt => ({
                label: vt.replace('_', ' ').charAt(0).toUpperCase() + vt.replace('_', ' ').slice(1),
                value: vt
              }))}
              disabled={loading}
            />
          </div>
        </Col>

        <Col xs={24}>
          <div className="filter-group">
            <label>Reduction Method</label>
            <Select
              value={filters.method}
              onChange={handleMethodChange}
              options={options.methods.map(m => {
                const uppercase_methods = ['umap', 'pacmap', 'pca'];
                const label = uppercase_methods.includes(m.toLowerCase())
                  ? m.toUpperCase()
                  : m.charAt(0).toUpperCase() + m.slice(1);
                return { label, value: m };
              })}
              disabled={loading}
            />
          </div>
        </Col>

        <Col xs={24}>
          <div className="filter-group">
            <label>Channel</label>
            <Select
              value={filters.channel}
              onChange={handleChannelChange}
              options={options.channels.map(c => ({
                label: c,
                value: c
              }))}
              disabled={loading}
              showSearch
            />
          </div>
        </Col>

        <Col xs={24}>
          <div className="filter-group">
            <label>Author</label>
            <Select
              value={filters.author}
              onChange={handleAuthorChange}
              options={options.authors.map(a => ({
                label: a,
                value: a
              }))}
              disabled={loading}
              showSearch
            />
          </div>
        </Col>

        <Col xs={24}>
          <div className="filter-group">
            <label>Cluster</label>
            <Select
              value={filters.cluster}
              onChange={handleClusterChange}
              options={options.clusters.map(c => ({
                label: c,
                value: c
              }))}
              disabled={loading}
              showSearch
            />
          </div>
        </Col>

        <Col xs={24}>
          <div className="filter-group">
            <label>Min Messages: {filters.min_messages}</label>
            <Slider
              min={1}
              max={50}
              step={null}
              marks={{
                1: '1',
                5: '5',
                10: '10',
                20: '20',
                30: '30',
                40: '40',
                50: '50',
              }}
              value={filters.min_messages}
              onChange={handleMinMessagesChange}
              disabled={loading}
            />
          </div>
        </Col>

        <Col xs={24}>
          <div className="filter-group">
            <label>Search descriptions</label>
            <Input
              placeholder="Search in topic, technical topic, sentiment..."
              value={filters.search}
              onChange={handleSearchChange}
              disabled={loading}
            />
          </div>
        </Col>
      </Row>
    </div>
  );
};
