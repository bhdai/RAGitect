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
