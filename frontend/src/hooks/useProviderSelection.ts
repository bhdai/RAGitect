'use client';

import { useState, useEffect, useCallback } from 'react';
import { getLLMConfigs, type LLMProviderConfig } from '@/lib/llmConfig';
import { LLM_PROVIDER_REGISTRY } from '@/lib/providers';

const STORAGE_KEY = 'ragitect-selected-provider';

export interface ProviderOption {
  providerName: string;
  displayName: string;
  model: string;
  isActive: boolean;
}

export function useProviderSelection() {
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [providers, setProviders] = useState<ProviderOption[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Load available providers on mount
  useEffect(() => {
    const loadProviders = async () => {
      try {
        const response = await getLLMConfigs();

        // Map to display format with model names
        const activeProviders = response.configs
          .filter(c => c.isActive)
          .map(config => {
            const registry = LLM_PROVIDER_REGISTRY[config.providerName];
            return {
              providerName: config.providerName,
              displayName: registry?.displayName || config.providerName,
              model: config.model || registry?.defaultModel || 'Unknown',
              isActive: config.isActive,
            };
          });

        setProviders(activeProviders);

        // Load saved selection or use first provider
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved && activeProviders.some(p => p.providerName === saved)) {
          setSelectedProvider(saved);
        } else if (activeProviders.length > 0) {
          setSelectedProvider(activeProviders[0].providerName);
        }
      } catch (error) {
        console.error('Failed to load providers:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadProviders();
  }, []);

  const selectProvider = useCallback((providerName: string) => {
    setSelectedProvider(providerName);
    localStorage.setItem(STORAGE_KEY, providerName);
  }, []);

  // Get current selection details
  const currentProvider = providers.find(p => p.providerName === selectedProvider);

  return {
    selectedProvider,
    selectProvider,
    providers,
    currentProvider,
    isLoading,
  };
}
