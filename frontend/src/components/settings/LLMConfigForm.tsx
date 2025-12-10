/**
 * LLM Configuration Form Component
 * 
 * Story 1.4: LLM Provider Configuration (Ollama & API Keys)
 * 
 * Allows users to configure LLM providers (Ollama, OpenAI, Anthropic)
 * with validation and test connection functionality.
 */

'use client';

import { useState, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { 
  CheckCircle2, 
  XCircle, 
  Loader2, 
  Server, 
  Key,
  Eye,
  EyeOff
} from 'lucide-react';
import { toast } from 'sonner';
import {
  type LLMProviderConfig,
  getLLMConfigs,
  saveLLMConfig,
  validateLLMConfig,
} from '@/lib/llmConfig';

// Provider configuration state
interface ProviderState {
  enabled: boolean;
  baseUrl: string;
  apiKey: string;
  model: string;
  isValidating: boolean;
  validationStatus: 'idle' | 'success' | 'error';
  validationMessage: string;
  isSaving: boolean;
  hasChanges: boolean;
  showApiKey: boolean;
}

const defaultOllamaState: ProviderState = {
  enabled: false,
  baseUrl: 'http://localhost:11434',
  apiKey: '',
  model: 'llama3.1:8b',
  isValidating: false,
  validationStatus: 'idle',
  validationMessage: '',
  isSaving: false,
  hasChanges: false,
  showApiKey: false,
};

const defaultOpenAIState: ProviderState = {
  enabled: false,
  baseUrl: '',
  apiKey: '',
  model: 'gpt-4o',
  isValidating: false,
  validationStatus: 'idle',
  validationMessage: '',
  isSaving: false,
  hasChanges: false,
  showApiKey: false,
};

const defaultAnthropicState: ProviderState = {
  enabled: false,
  baseUrl: '',
  apiKey: '',
  model: 'claude-3-5-sonnet-20241022',
  isValidating: false,
  validationStatus: 'idle',
  validationMessage: '',
  isSaving: false,
  hasChanges: false,
  showApiKey: false,
};

export function LLMConfigForm() {
  const [isLoading, setIsLoading] = useState(true);
  const [ollama, setOllama] = useState<ProviderState>(defaultOllamaState);
  const [openai, setOpenai] = useState<ProviderState>(defaultOpenAIState);
  const [anthropic, setAnthropic] = useState<ProviderState>(defaultAnthropicState);

  // Load existing configurations on mount
  useEffect(() => {
    const loadConfigs = async () => {
      try {
        const response = await getLLMConfigs();
        
        for (const config of response.configs) {
          const baseState = {
            enabled: config.isActive,
            model: config.model || '',
            baseUrl: config.baseUrl || '',
            apiKey: '', // Never expose API keys
          };

          switch (config.providerName) {
            case 'ollama':
              setOllama(prev => ({
                ...prev,
                ...baseState,
                baseUrl: config.baseUrl || defaultOllamaState.baseUrl,
              }));
              break;
            case 'openai':
              setOpenai(prev => ({
                ...prev,
                ...baseState,
                model: config.model || defaultOpenAIState.model,
              }));
              break;
            case 'anthropic':
              setAnthropic(prev => ({
                ...prev,
                ...baseState,
                model: config.model || defaultAnthropicState.model,
              }));
              break;
          }
        }
      } catch (error) {
        console.error('Failed to load LLM configs:', error);
        toast.error('Failed to load LLM configurations');
      } finally {
        setIsLoading(false);
      }
    };

    loadConfigs();
  }, []);

  // Generic handler for testing connection
  const handleTestConnection = useCallback(async (
    provider: 'ollama' | 'openai' | 'anthropic',
    state: ProviderState,
    setState: React.Dispatch<React.SetStateAction<ProviderState>>
  ) => {
    setState(prev => ({ 
      ...prev, 
      isValidating: true, 
      validationStatus: 'idle',
      validationMessage: '',
    }));

    try {
      const validateData = {
        providerName: provider,
        baseUrl: provider === 'ollama' ? state.baseUrl : undefined,
        apiKey: provider !== 'ollama' ? state.apiKey : undefined,
        model: state.model || undefined,
      };

      const result = await validateLLMConfig(validateData);

      setState(prev => ({
        ...prev,
        isValidating: false,
        validationStatus: result.valid ? 'success' : 'error',
        validationMessage: result.message,
      }));

      if (result.valid) {
        toast.success(result.message);
      } else {
        toast.error(result.error || result.message);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Validation failed';
      setState(prev => ({
        ...prev,
        isValidating: false,
        validationStatus: 'error',
        validationMessage: message,
      }));
      toast.error(message);
    }
  }, []);

  // Generic handler for saving configuration
  const handleSaveConfig = useCallback(async (
    provider: 'ollama' | 'openai' | 'anthropic',
    state: ProviderState,
    setState: React.Dispatch<React.SetStateAction<ProviderState>>
  ) => {
    setState(prev => ({ ...prev, isSaving: true }));

    try {
      const saveData = {
        providerName: provider,
        isActive: state.enabled,
        model: state.model || undefined,
        baseUrl: provider === 'ollama' ? state.baseUrl : undefined,
        apiKey: provider !== 'ollama' && state.apiKey ? state.apiKey : undefined,
      };

      await saveLLMConfig(saveData);

      setState(prev => ({ 
        ...prev, 
        isSaving: false, 
        hasChanges: false,
        apiKey: '', // Clear API key after save for security
      }));
      toast.success(`${provider.charAt(0).toUpperCase() + provider.slice(1)} configuration saved`);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to save configuration';
      setState(prev => ({ ...prev, isSaving: false }));
      toast.error(message);
    }
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-zinc-400" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Ollama Configuration */}
      <ProviderCard
        title="Ollama"
        description="Connect to a local Ollama instance for privacy-first LLM inference"
        icon={<Server className="h-5 w-5" />}
        provider="ollama"
        state={ollama}
        setState={setOllama}
        onTestConnection={() => handleTestConnection('ollama', ollama, setOllama)}
        onSave={() => handleSaveConfig('ollama', ollama, setOllama)}
      >
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="ollama-url">Base URL</Label>
            <Input
              id="ollama-url"
              type="url"
              placeholder="http://localhost:11434"
              value={ollama.baseUrl}
              onChange={(e) => setOllama(prev => ({ 
                ...prev, 
                baseUrl: e.target.value,
                hasChanges: true,
              }))}
              disabled={!ollama.enabled}
            />
            <p className="text-xs text-zinc-500">
              The base URL of your Ollama instance. Default: http://localhost:11434
            </p>
          </div>
          <div className="space-y-2">
            <Label htmlFor="ollama-model">Model</Label>
            <Input
              id="ollama-model"
              type="text"
              placeholder="llama3.1:8b"
              value={ollama.model}
              onChange={(e) => setOllama(prev => ({ 
                ...prev, 
                model: e.target.value,
                hasChanges: true,
              }))}
              disabled={!ollama.enabled}
            />
            <p className="text-xs text-zinc-500">
              The model to use (e.g., llama3.1:8b, mistral, codellama)
            </p>
          </div>
        </div>
      </ProviderCard>

      {/* OpenAI Configuration */}
      <ProviderCard
        title="OpenAI"
        description="Connect to OpenAI's API for GPT models"
        icon={<Key className="h-5 w-5" />}
        provider="openai"
        state={openai}
        setState={setOpenai}
        onTestConnection={() => handleTestConnection('openai', openai, setOpenai)}
        onSave={() => handleSaveConfig('openai', openai, setOpenai)}
      >
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="openai-key">API Key</Label>
            <div className="relative">
              <Input
                id="openai-key"
                type={openai.showApiKey ? 'text' : 'password'}
                placeholder="sk-..."
                value={openai.apiKey}
                onChange={(e) => setOpenai(prev => ({ 
                  ...prev, 
                  apiKey: e.target.value,
                  hasChanges: true,
                }))}
                disabled={!openai.enabled}
                className="pr-10"
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                onClick={() => setOpenai(prev => ({ ...prev, showApiKey: !prev.showApiKey }))}
                disabled={!openai.enabled}
              >
                {openai.showApiKey ? (
                  <EyeOff className="h-4 w-4 text-zinc-500" />
                ) : (
                  <Eye className="h-4 w-4 text-zinc-500" />
                )}
              </Button>
            </div>
            <p className="text-xs text-zinc-500">
              Your OpenAI API key starting with &quot;sk-&quot;
            </p>
          </div>
          <div className="space-y-2">
            <Label htmlFor="openai-model">Model</Label>
            <Input
              id="openai-model"
              type="text"
              placeholder="gpt-4o"
              value={openai.model}
              onChange={(e) => setOpenai(prev => ({ 
                ...prev, 
                model: e.target.value,
                hasChanges: true,
              }))}
              disabled={!openai.enabled}
            />
            <p className="text-xs text-zinc-500">
              The model to use (e.g., gpt-4o, gpt-4-turbo, gpt-3.5-turbo)
            </p>
          </div>
        </div>
      </ProviderCard>

      {/* Anthropic Configuration */}
      <ProviderCard
        title="Anthropic"
        description="Connect to Anthropic's API for Claude models"
        icon={<Key className="h-5 w-5" />}
        provider="anthropic"
        state={anthropic}
        setState={setAnthropic}
        onTestConnection={() => handleTestConnection('anthropic', anthropic, setAnthropic)}
        onSave={() => handleSaveConfig('anthropic', anthropic, setAnthropic)}
      >
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="anthropic-key">API Key</Label>
            <div className="relative">
              <Input
                id="anthropic-key"
                type={anthropic.showApiKey ? 'text' : 'password'}
                placeholder="sk-ant-..."
                value={anthropic.apiKey}
                onChange={(e) => setAnthropic(prev => ({ 
                  ...prev, 
                  apiKey: e.target.value,
                  hasChanges: true,
                }))}
                disabled={!anthropic.enabled}
                className="pr-10"
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                onClick={() => setAnthropic(prev => ({ ...prev, showApiKey: !prev.showApiKey }))}
                disabled={!anthropic.enabled}
              >
                {anthropic.showApiKey ? (
                  <EyeOff className="h-4 w-4 text-zinc-500" />
                ) : (
                  <Eye className="h-4 w-4 text-zinc-500" />
                )}
              </Button>
            </div>
            <p className="text-xs text-zinc-500">
              Your Anthropic API key starting with &quot;sk-ant-&quot;
            </p>
          </div>
          <div className="space-y-2">
            <Label htmlFor="anthropic-model">Model</Label>
            <Input
              id="anthropic-model"
              type="text"
              placeholder="claude-3-5-sonnet-20241022"
              value={anthropic.model}
              onChange={(e) => setAnthropic(prev => ({ 
                ...prev, 
                model: e.target.value,
                hasChanges: true,
              }))}
              disabled={!anthropic.enabled}
            />
            <p className="text-xs text-zinc-500">
              The model to use (e.g., claude-3-5-sonnet-20241022, claude-3-opus-20240229)
            </p>
          </div>
        </div>
      </ProviderCard>
    </div>
  );
}

// Provider Card component
interface ProviderCardProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  provider: 'ollama' | 'openai' | 'anthropic';
  state: ProviderState;
  setState: React.Dispatch<React.SetStateAction<ProviderState>>;
  onTestConnection: () => void;
  onSave: () => void;
  children: React.ReactNode;
}

function ProviderCard({
  title,
  description,
  icon,
  provider,
  state,
  setState,
  onTestConnection,
  onSave,
  children,
}: ProviderCardProps) {
  const canTest = provider === 'ollama' 
    ? state.baseUrl.trim() !== ''
    : state.apiKey.trim() !== '';

  return (
    <Card className={state.enabled ? '' : 'opacity-75'}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-zinc-100 dark:bg-zinc-800">
              {icon}
            </div>
            <div>
              <CardTitle className="flex items-center gap-2">
                {title}
                {state.enabled && (
                  <Badge variant="secondary" className="text-xs">
                    Enabled
                  </Badge>
                )}
              </CardTitle>
              <CardDescription>{description}</CardDescription>
            </div>
          </div>
          <Switch
            checked={state.enabled}
            onCheckedChange={(checked) => setState(prev => ({
              ...prev,
              enabled: checked,
              hasChanges: true,
            }))}
          />
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {children}
        
        <Separator />
        
        {/* Validation Status */}
        {state.validationStatus !== 'idle' && (
          <div className={`flex items-center gap-2 text-sm ${
            state.validationStatus === 'success' 
              ? 'text-green-600 dark:text-green-400' 
              : 'text-red-600 dark:text-red-400'
          }`}>
            {state.validationStatus === 'success' ? (
              <CheckCircle2 className="h-4 w-4" />
            ) : (
              <XCircle className="h-4 w-4" />
            )}
            {state.validationMessage}
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={onTestConnection}
            disabled={!state.enabled || !canTest || state.isValidating}
          >
            {state.isValidating ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Testing...
              </>
            ) : (
              'Test Connection'
            )}
          </Button>
          <Button
            size="sm"
            onClick={onSave}
            disabled={!state.hasChanges || state.isSaving}
          >
            {state.isSaving ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Saving...
              </>
            ) : (
              'Save Configuration'
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
