import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import type { Chunk, FilterState } from '../types';
import './ScatterChart.css';

interface ScatterChartProps {
  chunks: Chunk[];
  filters: FilterState;
  onChunkClick: (chunk: Chunk) => void;
  loading: boolean;
}

export const ScatterChart: React.FC<ScatterChartProps> = ({
  chunks,
  filters,
  onChunkClick,
  loading,
}) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<echarts.ECharts | null>(null);
  const [showLoading, setShowLoading] = React.useState(false);
  const loadingTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  // Only show loading after 300ms of loading state
  React.useEffect(() => {
    if (loading) {
      loadingTimeoutRef.current = setTimeout(() => {
        setShowLoading(true);
      }, 300);
    } else {
      if (loadingTimeoutRef.current) {
        clearTimeout(loadingTimeoutRef.current);
      }
      setShowLoading(false);
    }

    return () => {
      if (loadingTimeoutRef.current) {
        clearTimeout(loadingTimeoutRef.current);
      }
    };
  }, [loading]);

  useEffect(() => {
    if (!chartRef.current) return;

    // Initialize chart
    if (!chartInstanceRef.current) {
      chartInstanceRef.current = echarts.init(chartRef.current);
    }

    // Prepare scatter data
    const scatterData = chunks.map(chunk => ({
      value: [
        chunk.x,
        chunk.y,
        chunk.id,
        chunk.channel,
        chunk.topic,
        chunk.technical_topic,
        chunk.sentiment,
        chunk.messages,
        chunk.authors,
      ],
      itemStyle: {
        color: chunk.color,
        opacity: chunk.opacity,
      },
      symbolSize: 5,
    }));

    const option: echarts.EChartsOption = {
      backgroundColor: '#ffffff',
      textStyle: { color: '#333333' },
      title: {
        text: `${filters.vector_type.replace('_', ' ').toUpperCase()} - ${filters.method.toUpperCase()} (${chunks.length} chunks)`,
        textStyle: { color: '#000000', fontSize: 16 },
        left: 'center',
        top: 15,
      },
      tooltip: {
        trigger: 'item',
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        borderColor: '#d0d0d0',
        borderWidth: 1,
        textStyle: { color: '#333333', fontSize: 12 },
        confine: true,
        padding: [10, 15],
        formatter: (params: any) => {
          if (!params.value) return '';
          const v = params.value;
          return `
            <div style="line-height: 1.6;">
              <strong style="color: #1890ff;">Chunk ${v[2]}</strong><br/>
              <span style="color: #666666;">Channel: ${v[3]}</span><br/>
              <span style="color: #666666;">Topic: ${v[4]}</span><br/>
              <span style="color: #666666;">Technical: ${v[5]}</span><br/>
              <span style="color: #666666;">Sentiment: ${v[6]}</span><br/>
              <span style="color: #666666;">Messages: ${v[7]}</span><br/>
              <span style="color: #666666;">Authors: ${v[8]}</span>
            </div>
          `;
        },
      },
      grid: { left: '10%', right: '10%', top: '15%', bottom: '10%' },
      xAxis: {
        type: 'value',
        axisLine: { lineStyle: { color: '#d0d0d0' } },
        axisLabel: { show: false },
        splitLine: { lineStyle: { color: '#e8e8e8' } },
      },
      yAxis: {
        type: 'value',
        axisLine: { lineStyle: { color: '#d0d0d0' } },
        axisLabel: { show: false },
        splitLine: { lineStyle: { color: '#e8e8e8' } },
      },
      series: [
        {
          type: 'scatter',
          name: 'Chunks',
          symbolSize: 5,
          data: scatterData,
          itemStyle: { opacity: 0.6 },
        },
      ],
      animation: false,
    };

    chartInstanceRef.current.setOption(option);

    // Handle click events
    const handleChartClick = (params: any) => {
      if (params.componentSubType === 'scatter' && params.value) {
        const chunkId = params.value[2];
        const chunk = chunks.find(c => c.id === chunkId);
        if (chunk) {
          onChunkClick(chunk);
        }
      }
    };

    chartInstanceRef.current.off('click');
    chartInstanceRef.current.on('click', handleChartClick);

    // Handle window resize
    const handleResize = () => {
      chartInstanceRef.current?.resize();
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [chunks, filters, onChunkClick]);

  return (
    <div className="scatter-chart-container">
      <div ref={chartRef} className="scatter-chart" />
      {showLoading && <div className="chart-loading">Loading...</div>}
    </div>
  );
};
