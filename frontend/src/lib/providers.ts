export interface LLMProviderDefinition {
  id: string;
  displayName: string;
  icon: string;
  defaultBaseUrl?: string;
  requiresApiKey: boolean;
  apiKeyPrefix?: string;
  apiKeyPlaceholder?: string;
  defaultModel: string;
  popularModels: string[];
  description: string;
}

export const LLM_PROVIDER_REGISTRY: Record<string, LLMProviderDefinition> = {
  ollama: {
    id: 'ollama',
    displayName: 'Ollama',
    icon: 'Server',
    defaultBaseUrl: 'http://localhost:11434',
    requiresApiKey: false,
    defaultModel: 'llama3.1:8b',
    popularModels: [
      'llama3.1:8b',
      'llama3.1:70b',
      'llama3.2:3b',
      'llama3.2:1b',
      'mistral',
      'mixtral:8x7b',
      'codellama',
      'gemma2',
      'qwen2.5',
      'phi3',
      'deepseek-coder'
    ],
    description: 'Local LLM inference with complete privacy'
  },
  openai: {
    id: 'openai',
    displayName: 'OpenAI',
    icon: 'Key',
    defaultBaseUrl: 'https://api.openai.com/v1',
    requiresApiKey: true,
    apiKeyPrefix: 'sk-',
    apiKeyPlaceholder: 'sk-...',
    defaultModel: 'gpt-4o',
    popularModels: [
      'gpt-4o',
      'gpt-4o-mini',
      'gpt-4-turbo',
      'gpt-4',
      'gpt-3.5-turbo',
      'o1-preview',
      'o1-mini',
      'o3-mini'
    ],
    description: 'OpenAI GPT models via API'
  },
  anthropic: {
    id: 'anthropic',
    displayName: 'Anthropic',
    icon: 'Key',
    defaultBaseUrl: 'https://api.anthropic.com',
    requiresApiKey: true,
    apiKeyPrefix: 'sk-ant-',
    apiKeyPlaceholder: 'sk-ant-...',
    defaultModel: 'claude-sonnet-4-20250514',
    popularModels: [
      'claude-sonnet-4-20250514',
      'claude-3-5-sonnet-20241022',
      'claude-3-5-haiku-20241022',
      'claude-3-opus-20240229'
    ],
    description: 'Anthropic Claude models via API'
  },
  gemini: {
    id: 'gemini',
    displayName: 'Google Gemini',
    icon: 'Sparkles',
    defaultBaseUrl: 'https://generativelanguage.googleapis.com/v1beta',
    requiresApiKey: true,
    apiKeyPlaceholder: 'AI...',
    defaultModel: 'gemini-2.0-flash',
    popularModels: [
      'gemini-2.0-flash',
      'gemini-2.5-pro-preview-06-05',
      'gemini-1.5-pro',
      'gemini-1.5-flash',
      'gemini-1.5-flash-8b'
    ],
    description: 'Google Gemini models via API'
  },
  openai_compatible: {
    id: 'openai_compatible',
    displayName: 'OpenAI Compatible',
    icon: 'Plug',
    defaultBaseUrl: '',
    requiresApiKey: true,
    apiKeyPlaceholder: 'API key...',
    defaultModel: '',
    popularModels: [],
    description: 'Any OpenAI-compatible API (vLLM, LMStudio, etc.)'
  }
};

export const LLM_PROVIDER_OPTIONS = Object.values(LLM_PROVIDER_REGISTRY);

// Embedding Provider Definitions
export interface EmbeddingProviderDefinition {
  id: string;
  displayName: string;
  icon: string;
  defaultBaseUrl?: string;
  requiresApiKey: boolean;
  apiKeyPlaceholder?: string;
  defaultModel: string;
  popularModels: string[];
  defaultDimension: number;
  dimensionByModel: Record<string, number>;
  description: string;
}

export const EMBEDDING_PROVIDER_REGISTRY: Record<string, EmbeddingProviderDefinition> = {
  ollama: {
    id: 'ollama',
    displayName: 'Ollama',
    icon: 'Server',
    defaultBaseUrl: 'http://localhost:11434',
    requiresApiKey: false,
    defaultModel: 'nomic-embed-text',
    popularModels: [
      'nomic-embed-text',
      'mxbai-embed-large',
      'all-minilm',
      'snowflake-arctic-embed'
    ],
    defaultDimension: 768,
    dimensionByModel: {
      'nomic-embed-text': 768,
      'mxbai-embed-large': 1024,
      'all-minilm': 384,
      'snowflake-arctic-embed': 1024
    },
    description: 'Local embedding models with complete privacy'
  },
  openai: {
    id: 'openai',
    displayName: 'OpenAI',
    icon: 'Key',
    defaultBaseUrl: 'https://api.openai.com/v1',
    requiresApiKey: true,
    apiKeyPlaceholder: 'sk-...',
    defaultModel: 'text-embedding-3-small',
    popularModels: [
      'text-embedding-3-small',
      'text-embedding-3-large',
      'text-embedding-ada-002'
    ],
    defaultDimension: 1536,
    dimensionByModel: {
      'text-embedding-3-small': 1536,
      'text-embedding-3-large': 3072,
      'text-embedding-ada-002': 1536
    },
    description: 'OpenAI embedding models via API'
  },
  vertex_ai: {
    id: 'vertex_ai',
    displayName: 'Vertex AI',
    icon: 'Cloud',
    defaultBaseUrl: '',
    requiresApiKey: true,
    apiKeyPlaceholder: 'Google Cloud credentials...',
    defaultModel: 'text-embedding-004',
    popularModels: [
      'text-embedding-004',
      'text-multilingual-embedding-002'
    ],
    defaultDimension: 768,
    dimensionByModel: {
      'text-embedding-004': 768,
      'text-multilingual-embedding-002': 768
    },
    description: 'Google Vertex AI embedding models'
  },
  openai_compatible: {
    id: 'openai_compatible',
    displayName: 'OpenAI Compatible',
    icon: 'Plug',
    defaultBaseUrl: '',
    requiresApiKey: true,
    apiKeyPlaceholder: 'API key...',
    defaultModel: '',
    popularModels: [],
    defaultDimension: 768,
    dimensionByModel: {},
    description: 'Any OpenAI-compatible embedding API'
  }
};

export const EMBEDDING_PROVIDER_OPTIONS = Object.values(EMBEDDING_PROVIDER_REGISTRY);
