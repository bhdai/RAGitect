/**
 * LLM Configuration Form Component
 * 
 * Story 1.4: LLM Provider Configuration - Phase 1
 * 
 * Unified form with provider dropdown supporting:
 * - Ollama, OpenAI, Anthropic, Gemini
 * - Auto-fill base URL on provider selection
 * - Conditional API key field visibility
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { ModelCombobox } from '@/components/ui/model-combobox';
import { 
  CheckCircle2, 
  XCircle, 
  Loader2, 
  Server, 
  Key,
  Eye,
  EyeOff,
  Sparkles,
  Plug
} from 'lucide-react';
import { toast } from 'sonner';
import {
  type LLMProviderConfig,
  getLLMConfigs,
  saveLLMConfig,
  updateLLMConfig,
  validateLLMConfig,
  toggleLLMConfig,
} from '@/lib/llmConfig';
import { LLM_PROVIDER_REGISTRY, LLM_PROVIDER_OPTIONS } from '@/lib/providers';

// Unified form configuration state
interface ConfigFormState {
  selectedProvider: string;
  baseUrl: string;
  apiKey: string;
  model: string;
  isEnabled: boolean;
  isValidating: boolean;
  validationStatus: 'idle' | 'success' | 'error';
  validationMessage: string;
  isSaving: boolean;
  hasChanges: boolean;
  showApiKey: boolean;
}

const defaultFormState: ConfigFormState = {
  selectedProvider: 'ollama',
  baseUrl: 'http://localhost:11434',
  apiKey: '',
  model: 'llama3.1:8b',
  isEnabled: false,
  isValidating: false,
  validationStatus: 'idle',
  validationMessage: '',
  isSaving: false,
  hasChanges: false,
  showApiKey: false,
};

export function LLMConfigForm() {
  const [isLoading, setIsLoading] = useState(true);
  const [formState, setFormState] = useState<ConfigFormState>(defaultFormState);
  const [savedConfigs, setSavedConfigs] = useState<LLMProviderConfig[]>([]);

  // Get current provider definition
  const currentProvider = LLM_PROVIDER_REGISTRY[formState.selectedProvider];

  // Load existing configurations on mount
  useEffect(() => {
    const loadConfigs = async () => {
      try {
        const response = await getLLMConfigs();
        setSavedConfigs(response.configs);
        
        // Load the first active config, or default to ollama
        const activeConfig = response.configs.find(c => c.isActive) || response.configs[0];
        
        if (activeConfig) {
          const provider = LLM_PROVIDER_REGISTRY[activeConfig.providerName];
          setFormState({
            selectedProvider: activeConfig.providerName,
            baseUrl: activeConfig.baseUrl || provider?.defaultBaseUrl || '',
            model: activeConfig.model || provider?.defaultModel || '',
            apiKey: '', // Never expose API keys
            isEnabled: activeConfig.isActive,
            isValidating: false,
            validationStatus: 'idle',
            validationMessage: '',
            isSaving: false,
            hasChanges: false,
            showApiKey: false,
          });
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

  // Handle provider change with auto-fill
  const handleProviderChange = useCallback((providerId: string) => {
    const provider = LLM_PROVIDER_REGISTRY[providerId];
    const savedConfig = savedConfigs.find(c => c.providerName === providerId);
    
    setFormState(prev => ({
      ...prev,
      selectedProvider: providerId,
      baseUrl: savedConfig?.baseUrl || provider.defaultBaseUrl || '',
      model: savedConfig?.model || provider.defaultModel,
      apiKey: '', // Reset for security
      isEnabled: savedConfig?.isActive || false,
      hasChanges: true,
      validationStatus: 'idle',
      validationMessage: '',
    }));
  }, [savedConfigs]);

  // Handler for testing connection
  const handleTestConnection = useCallback(async () => {
    setFormState(prev => ({ 
      ...prev, 
      isValidating: true, 
      validationStatus: 'idle',
      validationMessage: '',
    }));

    try {
      const validateData = {
        providerName: formState.selectedProvider,
        baseUrl: !currentProvider.requiresApiKey ? formState.baseUrl : undefined,
        apiKey: currentProvider.requiresApiKey ? formState.apiKey : undefined,
        model: formState.model || undefined,
      };

      const result = await validateLLMConfig(validateData);

      setFormState(prev => ({
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
      setFormState(prev => ({
        ...prev,
        isValidating: false,
        validationStatus: 'error',
        validationMessage: message,
      }));
      toast.error(message);
    }
  }, [formState.selectedProvider, formState.baseUrl, formState.apiKey, formState.model, currentProvider]);

  // Handler for saving configuration
  const handleSaveConfig = useCallback(async () => {
    setFormState(prev => ({ ...prev, isSaving: true }));

    try {
      // Check if this provider already has a saved config
      const existingConfig = savedConfigs.find(
        c => c.providerName === formState.selectedProvider
      );

      if (existingConfig) {
        // Use PATCH for existing config - API key is optional
        const updateData: {
          model?: string;
          baseUrl?: string;
          apiKey?: string;
          isActive?: boolean;
        } = {
          isActive: formState.isEnabled,
        };

        if (formState.model) {
          updateData.model = formState.model;
        }

        if (!currentProvider.requiresApiKey && formState.baseUrl) {
          updateData.baseUrl = formState.baseUrl;
        }

        // Only include API key if user entered a new one
        if (currentProvider.requiresApiKey && formState.apiKey) {
          updateData.apiKey = formState.apiKey;
        }

        await updateLLMConfig(formState.selectedProvider, updateData);
      } else {
        // Use POST for new config - API key required for cloud providers
        const saveData = {
          providerName: formState.selectedProvider,
          isActive: formState.isEnabled,
          model: formState.model || undefined,
          baseUrl: !currentProvider.requiresApiKey ? formState.baseUrl : undefined,
          apiKey: currentProvider.requiresApiKey && formState.apiKey ? formState.apiKey : undefined,
        };

        await saveLLMConfig(saveData);
      }

      setFormState(prev => ({ 
        ...prev, 
        isSaving: false, 
        hasChanges: false,
        apiKey: '', // Clear API key after save for security
      }));
      toast.success(`${currentProvider.displayName} configuration saved`);
      
      // Reload configs
      const response = await getLLMConfigs();
      setSavedConfigs(response.configs);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to save configuration';
      setFormState(prev => ({ ...prev, isSaving: false }));
      toast.error(message);
    }
  }, [formState.selectedProvider, formState.isEnabled, formState.model, formState.baseUrl, formState.apiKey, currentProvider, savedConfigs]);

  // Handler for toggle switch - uses toggle endpoint for existing configs
  const handleToggle = useCallback(async (checked: boolean) => {
    // Check if this provider has a saved config
    const existingConfig = savedConfigs.find(
      c => c.providerName === formState.selectedProvider
    );

    if (existingConfig) {
      // Use toggle endpoint - no API key needed!
      try {
        await toggleLLMConfig(formState.selectedProvider, checked);
        setFormState(prev => ({ ...prev, isEnabled: checked }));
        toast.success(`${currentProvider.displayName} ${checked ? 'enabled' : 'disabled'}`);

        // Reload configs to sync state
        const response = await getLLMConfigs();
        setSavedConfigs(response.configs);
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to toggle provider';
        toast.error(message);
      }
    } else {
      // No saved config yet - just update form state (user needs to save first)
      setFormState(prev => ({
        ...prev,
        isEnabled: checked,
        hasChanges: true,
      }));
    }
  }, [formState.selectedProvider, savedConfigs, currentProvider]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-zinc-400" />
      </div>
    );
  }

  // Get icon component
  const getProviderIcon = (iconName: string) => {
    switch (iconName) {
      case 'Server':
        return <Server className="h-5 w-5" />;
      case 'Key':
        return <Key className="h-5 w-5" />;
      case 'Sparkles':
        return <Sparkles className="h-5 w-5" />;
      case 'Plug':
        return <Plug className="h-5 w-5" />;
      default:
        return <Server className="h-5 w-5" />;
    }
  };

  const canTest = currentProvider.requiresApiKey 
    ? formState.apiKey.trim() !== ''
    : formState.baseUrl.trim() !== '';

  return (
    <Card className={formState.isEnabled ? '' : 'opacity-75'}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          LLM Provider Configuration
          {formState.isEnabled && (
            <Badge variant="secondary" className="text-xs">
              Active
            </Badge>
          )}
        </CardTitle>
        <CardDescription>
          {currentProvider.description}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Provider Selection */}
        <div className="space-y-2">
          <Label htmlFor="provider">Provider</Label>
          <Select
            value={formState.selectedProvider}
            onValueChange={handleProviderChange}
          >
            <SelectTrigger id="provider">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {LLM_PROVIDER_OPTIONS.map((provider) => (
                <SelectItem key={provider.id} value={provider.id}>
                  <div className="flex items-center gap-2">
                    {getProviderIcon(provider.icon)}
                    <span>{provider.displayName}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-zinc-500">
            Choose your LLM provider
          </p>
        </div>

        {/* Base URL or API Key */}
        {currentProvider.requiresApiKey ? (
          <div className="space-y-2">
            <Label htmlFor="api-key">API Key</Label>
            <div className="relative">
              <Input
                id="api-key"
                type={formState.showApiKey ? 'text' : 'password'}
                placeholder={currentProvider.apiKeyPlaceholder}
                value={formState.apiKey}
                onChange={(e) => setFormState(prev => ({ 
                  ...prev, 
                  apiKey: e.target.value,
                  hasChanges: true,
                }))}
                disabled={!formState.isEnabled}
                className="pr-10"
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                onClick={() => setFormState(prev => ({ ...prev, showApiKey: !prev.showApiKey }))}
                disabled={!formState.isEnabled}
              >
                {formState.showApiKey ? (
                  <EyeOff className="h-4 w-4 text-zinc-500" />
                ) : (
                  <Eye className="h-4 w-4 text-zinc-500" />
                )}
              </Button>
            </div>
            <p className="text-xs text-zinc-500">
              Your {currentProvider.displayName} API key
              {currentProvider.apiKeyPrefix && ` starting with "${currentProvider.apiKeyPrefix}"`}
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            <Label htmlFor="base-url">Base URL</Label>
            <Input
              id="base-url"
              type="url"
              placeholder={currentProvider.defaultBaseUrl}
              value={formState.baseUrl}
              onChange={(e) => setFormState(prev => ({ 
                ...prev, 
                baseUrl: e.target.value,
                hasChanges: true,
              }))}
              disabled={!formState.isEnabled}
            />
            <p className="text-xs text-zinc-500">
              The base URL of your {currentProvider.displayName} instance
            </p>
          </div>
        )}

        {/* Model */}
        <div className="space-y-2">
          <Label htmlFor="model">Model</Label>
          <ModelCombobox
            value={formState.model}
            onChange={(model) => setFormState(prev => ({ 
              ...prev, 
              model,
              hasChanges: true,
            }))}
            options={currentProvider.popularModels}
            placeholder={currentProvider.defaultModel || "Select or enter model..."}
            disabled={!formState.isEnabled}
          />
          <p className="text-xs text-zinc-500">
            {currentProvider.popularModels.length > 0 
              ? "Select a popular model or type a custom name"
              : "Enter a custom model name"
            }
          </p>
        </div>

        {/* Active Toggle */}
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="is-active">Set as active</Label>
            <p className="text-xs text-zinc-500">
              Enable this provider for query processing
            </p>
          </div>
          <Switch
            id="is-active"
            checked={formState.isEnabled}
            onCheckedChange={handleToggle}
          />
        </div>

        <Separator />

        {/* Validation Status */}
        {formState.validationStatus !== 'idle' && (
          <div className={`flex items-center gap-2 text-sm ${
            formState.validationStatus === 'success' 
              ? 'text-green-600 dark:text-green-400' 
              : 'text-red-600 dark:text-red-400'
          }`}>
            {formState.validationStatus === 'success' ? (
              <CheckCircle2 className="h-4 w-4" />
            ) : (
              <XCircle className="h-4 w-4" />
            )}
            {formState.validationMessage}
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleTestConnection}
            disabled={!formState.isEnabled || !canTest || formState.isValidating}
          >
            {formState.isValidating ? (
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
            onClick={handleSaveConfig}
            disabled={!formState.hasChanges || formState.isSaving}
          >
            {formState.isSaving ? (
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

