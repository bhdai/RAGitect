'use client';

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { getLLMConfigs } from '@/lib/llmConfig';
import { LLM_PROVIDER_REGISTRY } from '@/lib/providers';

const STORAGE_KEY = 'ragitect-selected-provider';

export interface ProviderOption {
  providerName: string;
  displayName: string;
  model: string;
  isActive: boolean;
}

interface ProviderSelectionContextValue {
  selectedProvider: string | null;
  selectProvider: (providerName: string) => void;
  providers: ProviderOption[];
  currentProvider: ProviderOption | undefined;
  isLoading: boolean;
  error: Error | null;
}

const ProviderSelectionContext = createContext<ProviderSelectionContextValue | null>(null);

export function ProviderSelectionProvider({ children }: { children: ReactNode }) {
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [providers, setProviders] = useState<ProviderOption[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  // Load available providers on mount
  useEffect(() => {
    const loadProviders = async () => {
      try {
        setError(null);
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
      } catch (err) {
        const error = err instanceof Error ? err : new Error('Failed to load LLM providers');
        setError(error);
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

  return (
    <ProviderSelectionContext.Provider
      value={{
        selectedProvider,
        selectProvider,
        providers,
        currentProvider,
        isLoading,
        error,
      }}
    >
      {children}
    </ProviderSelectionContext.Provider>
  );
}

export function useProviderSelectionContext() {
  const context = useContext(ProviderSelectionContext);
  if (!context) {
    throw new Error('useProviderSelectionContext must be used within ProviderSelectionProvider');
  }
  return context;
}
