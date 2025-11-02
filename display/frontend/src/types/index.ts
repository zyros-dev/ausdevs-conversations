export interface Chunk {
  id: number;
  x: number;
  y: number;
  channel: string;
  topic: string;
  technical_topic: string;
  sentiment: string;
  messages: number;
  authors: string;
  color: string;
  opacity: number;
  cluster: number;
  start: string;
  end: string;
}

export interface Message {
  author: string;
  content: string;
  timestamp: string;
}

export interface ConversationDetail {
  chunk_id: number;
  channel: string;
  start: string;
  end: string;
  message_count: number;
  topic: string;
  technical_topic: string;
  sentiment: string;
  messages: Message[];
}

export interface FilterOptions {
  vector_types: string[];
  methods: string[];
  channels: string[];
  authors: string[];
  clusters: string[];
}

export interface FilterState {
  vector_type: string;
  method: string;
  channel: string;
  author: string;
  cluster: string;
  min_messages: number;
  search: string;
}
