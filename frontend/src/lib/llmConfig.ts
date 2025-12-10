/**
 * TypeScript types and API client for LLM Configuration
 * 
 * Story 1.4: LLM Provider Configuration (Ollama & API Keys)
 * Note: These match the camelCase serialization from the backend API
 */

// Types
export interface LLMProviderConfig {
  id: string;
  providerName: string;
  baseUrl?: string | null;
  model?: string | null;
  isActive: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface LLMProviderConfigListResponse {
  configs: LLMProviderConfig[];
  total: number;
}

export interface LLMProviderConfigCreateRequest {
  providerName: string;
  baseUrl?: string;
  apiKey?: string;
  model?: string;
  isActive?: boolean;
}

export interface LLMProviderConfigValidateRequest {
  providerName: string;
  baseUrl?: string;
  apiKey?: string;
  model?: string;
}

export interface LLMProviderConfigValidateResponse {
  valid: boolean;
  message: string;
  error?: string | null;
}

// API Client
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class LLMConfigApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: `Request failed with status ${response.status}`,
      }));
      throw new Error(error.detail);
    }

    // Handle 204 No Content
    if (response.status === 204) {
      return undefined as T;
    }

    return response.json();
  }

  /**
   * Fetch all LLM provider configurations
   */
  async getLLMConfigs(): Promise<LLMProviderConfigListResponse> {
    return this.request<LLMProviderConfigListResponse>('/api/v1/llm-configs');
  }

  /**
   * Fetch a specific provider configuration
   */
  async getLLMConfig(providerName: string): Promise<LLMProviderConfig> {
    return this.request<LLMProviderConfig>(`/api/v1/llm-configs/${providerName}`);
  }

  /**
   * Create or update LLM provider configuration
   */
  async saveLLMConfig(data: LLMProviderConfigCreateRequest): Promise<LLMProviderConfig> {
    return this.request<LLMProviderConfig>('/api/v1/llm-configs', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  /**
   * Delete LLM provider configuration
   */
  async deleteLLMConfig(providerName: string): Promise<void> {
    await this.request<void>(`/api/v1/llm-configs/${providerName}`, {
      method: 'DELETE',
    });
  }

  /**
   * Validate LLM provider configuration
   */
  async validateLLMConfig(
    data: LLMProviderConfigValidateRequest
  ): Promise<LLMProviderConfigValidateResponse> {
    return this.request<LLMProviderConfigValidateResponse>('/api/v1/llm-configs/validate', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }
}

// Export singleton instance
export const llmConfigApiClient = new LLMConfigApiClient();

// Export individual functions for convenience
export const getLLMConfigs = () => llmConfigApiClient.getLLMConfigs();
export const getLLMConfig = (providerName: string) => 
  llmConfigApiClient.getLLMConfig(providerName);
export const saveLLMConfig = (data: LLMProviderConfigCreateRequest) => 
  llmConfigApiClient.saveLLMConfig(data);
export const deleteLLMConfig = (providerName: string) => 
  llmConfigApiClient.deleteLLMConfig(providerName);
export const validateLLMConfig = (data: LLMProviderConfigValidateRequest) => 
  llmConfigApiClient.validateLLMConfig(data);
