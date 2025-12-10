/**
 * Embedding Configuration Form Component
 *
 * Story 1.4: LLM Provider Configuration - Phase 3
 *
 * Unified form with provider dropdown supporting:
 * - Ollama, OpenAI, Vertex AI, OpenAI Compatible
 * - Auto-fill base URL and dimension on provider/model selection
 * - Conditional API key field visibility
 * - Warning about dimension changes
 */

'use client';

import { useState, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  CheckCircle2,
  XCircle,
  Loader2,
  Server,
  Key,
  Eye,
  EyeOff,
  Cloud,
  Plug,
  AlertTriangle
} from 'lucide-react';
import { toast } from 'sonner';
import {
  type EmbeddingConfig,
  getEmbeddingConfigs,
  saveEmbeddingConfig,
  validateEmbeddingConfig,
} from '@/lib/llmConfig';
import { EMBEDDING_PROVIDER_REGISTRY, EMBEDDING_PROVIDER_OPTIONS } from '@/lib/providers';

// Unified form configuration state
interface ConfigFormState {
  selectedProvider: string;
  baseUrl: string;
  apiKey: string;
  model: string;
  dimension: number;
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
  model: 'nomic-embed-text',
  dimension: 768,
  isEnabled: false,
  isValidating: false,
  validationStatus: 'idle',
  validationMessage: '',
  isSaving: false,
  hasChanges: false,
  showApiKey: false,
};

export function EmbeddingConfigForm() {
  const [isLoading, setIsLoading] = useState(true);
  const [formState, setFormState] = useState<ConfigFormState>(defaultFormState);
  const [savedConfigs, setSavedConfigs] = useState<EmbeddingConfig[]>([]);

  // Get current provider definition
  const currentProvider = EMBEDDING_PROVIDER_REGISTRY[formState.selectedProvider];

  // Load existing configurations on mount
  useEffect(() => {
    const loadConfigs = async () => {
      try {
        const response = await getEmbeddingConfigs();
        setSavedConfigs(response.configs);

        // Load the first active config, or default to ollama
        const activeConfig = response.configs.find((c: EmbeddingConfig) => c.isActive) || response.configs[0];

        if (activeConfig) {
          const provider = EMBEDDING_PROVIDER_REGISTRY[activeConfig.providerName];
          setFormState({
            selectedProvider: activeConfig.providerName,
            baseUrl: activeConfig.baseUrl || provider?.defaultBaseUrl || '',
            model: activeConfig.model || provider?.defaultModel || '',
            dimension: activeConfig.dimension || provider?.defaultDimension || 768,
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
        console.error('Failed to load embedding configs:', error);
        toast.error('Failed to load embedding configurations');
      } finally {
        setIsLoading(false);
      }
    };

    loadConfigs();
  }, []);

  // Handle provider change with auto-fill
  const handleProviderChange = useCallback((providerId: string) => {
    const provider = EMBEDDING_PROVIDER_REGISTRY[providerId];
    const savedConfig = savedConfigs.find(c => c.providerName === providerId);

    setFormState(prev => ({
      ...prev,
      selectedProvider: providerId,
      baseUrl: savedConfig?.baseUrl || provider.defaultBaseUrl || '',
      model: savedConfig?.model || provider.defaultModel,
      dimension: savedConfig?.dimension || provider.defaultDimension,
      apiKey: '', // Reset for security
      isEnabled: savedConfig?.isActive || false,
      hasChanges: true,
      validationStatus: 'idle',
      validationMessage: '',
    }));
  }, [savedConfigs]);

  // Handle model change with auto-dimension
  const handleModelChange = useCallback((newModel: string) => {
    const dimension = currentProvider.dimensionByModel[newModel] || currentProvider.defaultDimension;
    setFormState(prev => ({
      ...prev,
      model: newModel,
      dimension,
      hasChanges: true,
      validationStatus: 'idle',
    }));
  }, [currentProvider]);

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

      const result = await validateEmbeddingConfig(validateData);

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
      const saveData = {
        providerName: formState.selectedProvider,
        isActive: formState.isEnabled,
        model: formState.model || undefined,
        dimension: formState.dimension,
        baseUrl: !currentProvider.requiresApiKey ? formState.baseUrl : undefined,
        apiKey: currentProvider.requiresApiKey && formState.apiKey ? formState.apiKey : undefined,
      };

      await saveEmbeddingConfig(saveData);

      setFormState(prev => ({
        ...prev,
        isSaving: false,
        hasChanges: false,
        apiKey: '', // Clear API key after save for security
      }));
      toast.success(`${currentProvider.displayName} embedding configuration saved`);

      // Reload configs
      const response = await getEmbeddingConfigs();
      setSavedConfigs(response.configs);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to save configuration';
      setFormState(prev => ({ ...prev, isSaving: false }));
      toast.error(message);
    }
  }, [formState.selectedProvider, formState.isEnabled, formState.model, formState.dimension, formState.baseUrl, formState.apiKey, currentProvider]);

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
      case 'Cloud':
        return <Cloud className="h-5 w-5" />;
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
          Embedding Model Configuration
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
        {/* Warning Banner */}
        <Alert variant="default" className="border-amber-500/50 bg-amber-500/10">
          <AlertTriangle className="h-4 w-4 text-amber-500" />
          <AlertDescription className="text-sm text-zinc-300">
            Changing embedding model will invalidate all existing search indices.
            You must re-process all documents in existing workspaces.
          </AlertDescription>
        </Alert>

        {/* Provider Selection */}
        <div className="space-y-2">
          <Label htmlFor="embedding-provider">Provider</Label>
          <Select
            value={formState.selectedProvider}
            onValueChange={handleProviderChange}
          >
            <SelectTrigger id="embedding-provider">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {EMBEDDING_PROVIDER_OPTIONS.map((provider) => (
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
            Choose your embedding provider
          </p>
        </div>

        {/* Base URL or API Key */}
        {currentProvider.requiresApiKey ? (
          <div className="space-y-2">
            <Label htmlFor="embedding-api-key">API Key</Label>
            <div className="relative">
              <Input
                id="embedding-api-key"
                type={formState.showApiKey ? 'text' : 'password'}
                placeholder={currentProvider.apiKeyPlaceholder}
                value={formState.apiKey}
                onChange={(e) => setFormState(prev => ({
                  ...prev,
                  apiKey: e.target.value,
                  hasChanges: true,
                  validationStatus: 'idle',
                }))}
                className="pr-10"
              />
              <button
                type="button"
                onClick={() => setFormState(prev => ({ ...prev, showApiKey: !prev.showApiKey }))}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-400 hover:text-zinc-300"
              >
                {formState.showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
            <p className="text-xs text-zinc-500">
              Your API key will be encrypted
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            <Label htmlFor="embedding-base-url">Base URL</Label>
            <Input
              id="embedding-base-url"
              type="url"
              placeholder="http://localhost:11434"
              value={formState.baseUrl}
              onChange={(e) => setFormState(prev => ({
                ...prev,
                baseUrl: e.target.value,
                hasChanges: true,
                validationStatus: 'idle',
              }))}
            />
            <p className="text-xs text-zinc-500">
              {currentProvider.id === 'ollama' ? 'Default: http://localhost:11434' : 'Provider endpoint URL'}
            </p>
          </div>
        )}

        {/* Model Selection */}
        <div className="space-y-2">
          <Label htmlFor="embedding-model">Model</Label>
          {currentProvider.popularModels.length > 0 ? (
            <Select
              value={formState.model}
              onValueChange={handleModelChange}
            >
              <SelectTrigger id="embedding-model">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {currentProvider.popularModels.map((model) => (
                  <SelectItem key={model} value={model}>
                    {model}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Input
              id="embedding-model"
              type="text"
              placeholder="Model name"
              value={formState.model}
              onChange={(e) => setFormState(prev => ({
                ...prev,
                model: e.target.value,
                hasChanges: true,
                validationStatus: 'idle',
              }))}
            />
          )}
          <p className="text-xs text-zinc-500">
            Choose embedding model
          </p>
        </div>

        {/* Dimension */}
        <div className="space-y-2">
          <Label htmlFor="embedding-dimension">Dimension</Label>
          <Input
            id="embedding-dimension"
            type="number"
            value={formState.dimension}
            onChange={(e) => setFormState(prev => ({
              ...prev,
              dimension: parseInt(e.target.value, 10) || prev.dimension,
              hasChanges: true,
              validationStatus: 'idle',
            }))}
            min={1}
            step={1}
          />
          <p className="text-xs text-zinc-500">
            Auto-detected based on model
          </p>
        </div>

        {/* Validation Status */}
        {formState.validationStatus !== 'idle' && (
          <div className={`flex items-center gap-2 text-sm ${
            formState.validationStatus === 'success' ? 'text-green-500' : 'text-red-500'
          }`}>
            {formState.validationStatus === 'success' ? (
              <CheckCircle2 className="h-4 w-4" />
            ) : (
              <XCircle className="h-4 w-4" />
            )}
            <span>{formState.validationMessage}</span>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex gap-2 pt-2">
          <Button
            variant="outline"
            onClick={handleTestConnection}
            disabled={!canTest || formState.isValidating}
          >
            {formState.isValidating ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Testing...
              </>
            ) : (
              'Test Embedding'
            )}
          </Button>
          <Button
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
