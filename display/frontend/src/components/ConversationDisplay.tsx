import React, { useState } from 'react';
import { Spin, Empty, Collapse } from 'antd';
import type { ConversationDetail } from '../types';
import './ConversationDisplay.css';

interface ConversationDisplayProps {
  conversation: ConversationDetail | null;
  loading: boolean;
}

export const ConversationDisplay: React.FC<ConversationDisplayProps> = ({
  conversation,
  loading,
}) => {
  const [analysisExpanded, setAnalysisExpanded] = useState(false);
  if (loading) {
    return (
      <div className="conversation-container">
        <Spin />
      </div>
    );
  }

  if (!conversation) {
    return (
      <div className="conversation-container">
        <Empty description="Click on a point to see the conversation" />
      </div>
    );
  }

  return (
    <div className="conversation-container">
      <div className="conversation-header">
        <h2>Conversation</h2>
        <div className="conversation-metadata">
          <div>
            <strong>Channel:</strong> {conversation.channel}
          </div>
          <div>
            <strong>Time:</strong> {conversation.start} → {conversation.end}
          </div>
          <div>
            <strong>Messages:</strong> {conversation.message_count}
          </div>
        </div>
      </div>

      <Collapse
        items={[
          {
            key: 'analysis',
            label: 'Analysis',
            children: (
              <div className="descriptions-section">
                <div className="description-group">
                  <h4>Topic</h4>
                  <p>{conversation.topic}</p>
                </div>
                <div className="description-group">
                  <h4>Technical Topic</h4>
                  <p>{conversation.technical_topic}</p>
                </div>
                <div className="description-group">
                  <h4>Sentiment</h4>
                  <p>{conversation.sentiment}</p>
                </div>
              </div>
            ),
          },
        ]}
        className="analysis-collapse"
      />

      <div className="messages-list">
        {conversation.messages.map((msg, idx) => (
          <div key={idx} className="message">
            <div className="message-header">
              <span className="message-author">{msg.author}</span>
              <span className="message-timestamp">{msg.timestamp}</span>
            </div>
            <div className="message-content">{msg.content}</div>
          </div>
        ))}
      </div>
    </div>
  );
};
