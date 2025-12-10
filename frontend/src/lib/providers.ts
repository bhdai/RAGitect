export interface LLMProviderDefinition {
  id: string;
  displayName: string;
  icon: string;
  defaultBaseUrl?: string;
  requiresApiKey: boolean;
  apiKeyPrefix?: string;
  apiKeyPlaceholder?: string;
  defaultModel: string;
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
    description: 'Google Gemini models via API'
  }
};

export const LLM_PROVIDER_OPTIONS = Object.values(LLM_PROVIDER_REGISTRY);
