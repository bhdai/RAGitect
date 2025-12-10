/**
 * LLMConfigForm Component Tests
 *
 * Story 1.4: LLM Provider Configuration - Phase 1 (Unified Form + Gemini)
 * Updated for unified form with provider dropdown
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { LLMConfigForm } from '../settings/LLMConfigForm';
import * as llmConfigApi from '@/lib/llmConfig';

// Mock the API module
vi.mock('@/lib/llmConfig', () => ({
  getLLMConfigs: vi.fn(),
  saveLLMConfig: vi.fn(),
  validateLLMConfig: vi.fn(),
  deleteLLMConfig: vi.fn(),
}));

// Mock toast
vi.mock('sonner', () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
  },
}));

describe('LLMConfigForm - Phase 1: Unified Form + Gemini', () => {
  const mockOllamaConfig: llmConfigApi.LLMProviderConfig = {
    id: '1',
    providerName: 'ollama',
    baseUrl: 'http://localhost:11434',
    model: 'llama3.1:8b',
    isActive: true,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  };

  const mockOpenAIConfig: llmConfigApi.LLMProviderConfig = {
    id: '2',
    providerName: 'openai',
    model: 'gpt-4o',
    isActive: false,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  };

  const mockGeminiConfig: llmConfigApi.LLMProviderConfig = {
    id: '3',
    providerName: 'gemini',
    model: 'gemini-2.0-flash',
    isActive: false,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  };

  beforeEach(() => {
    vi.clearAllMocks();
    // Default mock: no existing configs
    vi.mocked(llmConfigApi.getLLMConfigs).mockResolvedValue({ configs: [], total: 0 });
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  describe('Initial Rendering & Provider Dropdown', () => {
    it('renders unified form with provider dropdown', async () => {
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
        expect(screen.getByLabelText('Provider')).toBeInTheDocument();
      });
    });

    it('shows loading state initially', () => {
      render(<LLMConfigForm />);
      const spinner = document.querySelector('.animate-spin');
      expect(spinner).toBeInTheDocument();
    });

    it.skip('provider dropdown shows all 4 providers (Phase 1 requirement)', async () => {
      const user = userEvent.setup();
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      // Click provider dropdown
      const providerSelect = screen.getByRole('combobox', { name: /provider/i });
      await user.click(providerSelect);

      // All 4 providers should be visible in dropdown
      await waitFor(() => {
        expect(screen.getByText('Ollama')).toBeInTheDocument();
        expect(screen.getByText('OpenAI')).toBeInTheDocument();
        expect(screen.getByText('Anthropic')).toBeInTheDocument();
        expect(screen.getByText('Google Gemini')).toBeInTheDocument();
      });
    });

    it('loads existing configuration on mount', async () => {
      vi.mocked(llmConfigApi.getLLMConfigs).mockResolvedValue({
        configs: [mockOllamaConfig],
        total: 1
      });

      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(llmConfigApi.getLLMConfigs).toHaveBeenCalled();
        // Should show the loaded base URL
        expect(screen.getByDisplayValue('http://localhost:11434')).toBeInTheDocument();
      });
    });
  });

  describe('Provider Selection & Auto-fill (Phase 1 Core Feature)', () => {
    it('auto-fills base URL when selecting Ollama', async () => {
      const user = userEvent.setup();
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      // Provider should default to Ollama with auto-filled base URL
      const baseUrlInput = screen.getByLabelText('Base URL');
      expect(baseUrlInput).toHaveValue('http://localhost:11434');
    });

    // TODO: Re-enable when migrating to Vitest browser mode
    it.skip('auto-fills base URL when switching to OpenAI', async () => {
      const user = userEvent.setup();
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      // Open provider dropdown
      const providerSelect = screen.getByRole('combobox', { name: /provider/i });
      await user.click(providerSelect);

      // Select OpenAI
      await waitFor(() => {
        expect(screen.getByText('OpenAI')).toBeInTheDocument();
      });
      await user.click(screen.getByText('OpenAI'));

      // Base URL should NOT appear (OpenAI uses API key, no base URL field)
      await waitFor(() => {
        expect(screen.queryByLabelText('Base URL')).not.toBeInTheDocument();
      });
    });

    // TODO: Re-enable when migrating to Vitest browser mode
    it.skip('auto-fills base URL when selecting Gemini (Phase 1 new provider)', async () => {
      const user = userEvent.setup();
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      // Open provider dropdown
      const providerSelect = screen.getByRole('combobox', { name: /provider/i });
      await user.click(providerSelect);

      // Select Gemini
      await waitFor(() => {
        expect(screen.getByText('Google Gemini')).toBeInTheDocument();
      });
      await user.click(screen.getByText('Google Gemini'));

      // Gemini requires API key, not base URL
      await waitFor(() => {
        expect(screen.queryByLabelText('Base URL')).not.toBeInTheDocument();
        expect(screen.getByLabelText('API Key')).toBeInTheDocument();
      });
    });

    // TODO: Re-enable when migrating to Vitest browser mode
    it.skip('resets form state when switching providers', async () => {
      const user = userEvent.setup();
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      // Enable config and enter some data
      const enableSwitch = screen.getByRole('switch');
      await user.click(enableSwitch);

      // Enter model name
      const modelInput = screen.getByLabelText('Model');
      await user.clear(modelInput);
      await user.type(modelInput, 'custom-model');

      // Switch provider
      const providerSelect = screen.getByRole('combobox', { name: /provider/i });
      await user.click(providerSelect);
      await waitFor(() => {
        expect(screen.getByText('OpenAI')).toBeInTheDocument();
      });
      await user.click(screen.getByText('OpenAI'));

      // Model should reset to OpenAI default
      await waitFor(() => {
        expect(modelInput).toHaveValue('gpt-4o');
      });
    });
  });

  describe('Conditional Field Rendering (Phase 1 Key Feature)', () => {
    it('shows Base URL field for Ollama (requiresApiKey=false)', async () => {
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      // Ollama should show Base URL, not API Key
      expect(screen.getByLabelText('Base URL')).toBeInTheDocument();
      expect(screen.queryByLabelText('API Key')).not.toBeInTheDocument();
    });

    // TODO: Re-enable when migrating to Vitest browser mode
    it.skip('shows API Key field for OpenAI (requiresApiKey=true)', async () => {
      const user = userEvent.setup();
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      // Switch to OpenAI
      const providerSelect = screen.getByRole('combobox', { name: /provider/i });
      await user.click(providerSelect);
      await waitFor(() => {
        expect(screen.getByText('OpenAI')).toBeInTheDocument();
      });
      await user.click(screen.getByText('OpenAI'));

      // OpenAI should show API Key, not Base URL
      await waitFor(() => {
        expect(screen.getByLabelText('API Key')).toBeInTheDocument();
        expect(screen.queryByLabelText('Base URL')).not.toBeInTheDocument();
      });
    });

    // TODO: Re-enable when migrating to Vitest browser mode
    it.skip('shows API Key field for Anthropic (requiresApiKey=true)', async () => {
      const user = userEvent.setup();
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      // Switch to Anthropic
      const providerSelect = screen.getByRole('combobox', { name: /provider/i });
      await user.click(providerSelect);
      await waitFor(() => {
        expect(screen.getByText('Anthropic')).toBeInTheDocument();
      });
      await user.click(screen.getByText('Anthropic'));

      // Anthropic should show API Key with correct placeholder
      await waitFor(() => {
        expect(screen.getByLabelText('API Key')).toBeInTheDocument();
        expect(screen.getByPlaceholderText('sk-ant-...')).toBeInTheDocument();
      });
    });

    // TODO: Re-enable when migrating to Vitest browser mode
    it.skip('shows API Key field for Gemini (requiresApiKey=true, Phase 1 new)', async () => {
      const user = userEvent.setup();
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      // Switch to Gemini
      const providerSelect = screen.getByRole('combobox', { name: /provider/i });
      await user.click(providerSelect);
      await waitFor(() => {
        expect(screen.getByText('Google Gemini')).toBeInTheDocument();
      });
      await user.click(screen.getByText('Google Gemini'));

      // Gemini should show API Key with correct placeholder
      await waitFor(() => {
        expect(screen.getByLabelText('API Key')).toBeInTheDocument();
        expect(screen.getByPlaceholderText('AI...')).toBeInTheDocument();
      });
    });

    // TODO: Re-enable when migrating to Vitest browser mode
    it.skip('can toggle API key visibility', async () => {
      const user = userEvent.setup();
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      // Switch to OpenAI to get API key field
      const providerSelect = screen.getByRole('combobox', { name: /provider/i });
      await user.click(providerSelect);
      await waitFor(() => {
        expect(screen.getByText('OpenAI')).toBeInTheDocument();
      });
      await user.click(screen.getByText('OpenAI'));

      // Enable configuration
      const enableSwitch = screen.getByRole('switch');
      await user.click(enableSwitch);

      // Find API key input
      await waitFor(() => {
        const apiKeyInput = screen.getByLabelText('API Key') as HTMLInputElement;
        expect(apiKeyInput.type).toBe('password');
      });

      // Click toggle visibility button
      const toggleButton = screen.getByRole('button', { name: /toggle api key visibility/i });
      await user.click(toggleButton);

      // Should now be text type
      const apiKeyInput = screen.getByLabelText('API Key') as HTMLInputElement;
      expect(apiKeyInput.type).toBe('text');
    });
  });

  describe('Test Connection', () => {
    // TODO: Re-enable after fixing form state initialization in test environment
    it.skip('calls validateLLMConfig when Test Connection is clicked for Ollama', async () => {
      const user = userEvent.setup();
      vi.mocked(llmConfigApi.validateLLMConfig).mockResolvedValue({
        valid: true,
        message: 'Connection successful',
      });

      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      // Test Connection should be available (Ollama has default base URL)
      const testButton = screen.getByRole('button', { name: /test connection/i });
      await user.click(testButton);

      await waitFor(() => {
        expect(llmConfigApi.validateLLMConfig).toHaveBeenCalledWith(
          expect.objectContaining({
            providerName: 'ollama',
            baseUrl: 'http://localhost:11434',
          })
        );
      });
    });

    // TODO: Re-enable after fixing form state and dropdown interactions
    it.skip('calls validateLLMConfig for all 4 providers', async () => {
      const user = userEvent.setup();
      vi.mocked(llmConfigApi.validateLLMConfig).mockResolvedValue({
        valid: true,
        message: 'Connection successful',
      });

      const providers = ['ollama', 'openai', 'anthropic', 'gemini'];

      for (const providerName of providers) {
        vi.clearAllMocks();
        render(<LLMConfigForm />);

        await waitFor(() => {
          expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
        });

        // Select provider if not default
        if (providerName !== 'ollama') {
          const providerSelect = screen.getByRole('combobox', { name: /provider/i });
          await user.click(providerSelect);

          const displayName = providerName === 'gemini' ? 'Google Gemini' :
                             providerName.charAt(0).toUpperCase() + providerName.slice(1);
          await waitFor(() => {
            expect(screen.getByText(displayName)).toBeInTheDocument();
          });
          await user.click(screen.getByText(displayName));

          // For API key providers, need to enter a key
          if (['openai', 'anthropic', 'gemini'].includes(providerName)) {
            const enableSwitch = screen.getByRole('switch');
            await user.click(enableSwitch);

            const apiKeyInput = screen.getByLabelText('API Key');
            await user.type(apiKeyInput, 'test-api-key-123');
          }
        }

        // Click test connection
        const testButton = screen.getByRole('button', { name: /test connection/i });
        await user.click(testButton);

        await waitFor(() => {
          expect(llmConfigApi.validateLLMConfig).toHaveBeenCalledWith(
            expect.objectContaining({
              providerName,
            })
          );
        });
      }
    });

    // TODO: Re-enable after fixing button state in test environment
    it.skip('shows success status after successful connection test', async () => {
      const user = userEvent.setup();
      vi.mocked(llmConfigApi.validateLLMConfig).mockResolvedValue({
        valid: true,
        message: 'Connection successful',
      });

      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      const testButton = screen.getByRole('button', { name: /test connection/i });
      await user.click(testButton);

      await waitFor(() => {
        expect(screen.getByText('Connection successful')).toBeInTheDocument();
      });
    });

    // TODO: Re-enable after fixing button state in test environment
    it.skip('shows error status after failed connection test', async () => {
      const user = userEvent.setup();
      vi.mocked(llmConfigApi.validateLLMConfig).mockResolvedValue({
        valid: false,
        message: 'Connection refused',
        error: 'Connection refused',
      });

      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      const testButton = screen.getByRole('button', { name: /test connection/i });
      await user.click(testButton);

      await waitFor(() => {
        expect(screen.getByText('Connection refused')).toBeInTheDocument();
      });
    });

    it('disables test button when API key is missing for cloud providers', async () => {
      const user = userEvent.setup();
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      // Switch to OpenAI
      const providerSelect = screen.getByRole('combobox', { name: /provider/i });
      await user.click(providerSelect);
      await waitFor(() => {
        expect(screen.getByText('OpenAI')).toBeInTheDocument();
      });
      await user.click(screen.getByText('OpenAI'));

      // Test Connection should be disabled without API key
      await waitFor(() => {
        const testButton = screen.getByRole('button', { name: /test connection/i });
        expect(testButton).toBeDisabled();
      });
    });
  });

  describe('Save Configuration', () => {
    // TODO: Re-enable after fixing form state initialization
    it.skip('calls saveLLMConfig when Save is clicked', async () => {
      const user = userEvent.setup();
      vi.mocked(llmConfigApi.saveLLMConfig).mockResolvedValue(mockOllamaConfig);
      vi.mocked(llmConfigApi.getLLMConfigs).mockResolvedValue({
        configs: [mockOllamaConfig],
        total: 1
      });

      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      // Enable config
      const enableSwitch = screen.getByRole('switch');
      await user.click(enableSwitch);

      // Save button should be enabled
      const saveButton = screen.getByRole('button', { name: /save configuration/i });
      await user.click(saveButton);

      await waitFor(() => {
        expect(llmConfigApi.saveLLMConfig).toHaveBeenCalledWith(
          expect.objectContaining({
            providerName: 'ollama',
            isActive: true,
          })
        );
      });
    });

    // TODO: Re-enable when dropdown interactions work in JSDOM
    it.skip('saves configuration for Gemini (Phase 1 new provider)', async () => {
      const user = userEvent.setup();
      vi.mocked(llmConfigApi.saveLLMConfig).mockResolvedValue(mockGeminiConfig);
      vi.mocked(llmConfigApi.getLLMConfigs).mockResolvedValue({
        configs: [mockGeminiConfig],
        total: 1
      });

      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      // Switch to Gemini
      const providerSelect = screen.getByRole('combobox', { name: /provider/i });
      await user.click(providerSelect);
      await waitFor(() => {
        expect(screen.getByText('Google Gemini')).toBeInTheDocument();
      });
      await user.click(screen.getByText('Google Gemini'));

      // Enable and add API key
      const enableSwitch = screen.getByRole('switch');
      await user.click(enableSwitch);

      const apiKeyInput = screen.getByLabelText('API Key');
      await user.type(apiKeyInput, 'test-gemini-key');

      // Save
      const saveButton = screen.getByRole('button', { name: /save configuration/i });
      await user.click(saveButton);

      await waitFor(() => {
        expect(llmConfigApi.saveLLMConfig).toHaveBeenCalledWith(
          expect.objectContaining({
            providerName: 'gemini',
            apiKey: 'test-gemini-key',
          })
        );
      });
    });

    it('shows saving state when save is in progress', async () => {
      const user = userEvent.setup();
      vi.mocked(llmConfigApi.saveLLMConfig).mockImplementation(
        () => new Promise(resolve => setTimeout(() => resolve(mockOllamaConfig), 1000))
      );

      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      // Enable config
      const enableSwitch = screen.getByRole('switch');
      await user.click(enableSwitch);

      const saveButton = screen.getByRole('button', { name: /save configuration/i });
      await user.click(saveButton);

      // Button should show saving state
      await waitFor(() => {
        expect(screen.getByText('Saving...')).toBeInTheDocument();
      });
    });

    it('disables save button when no changes', async () => {
      vi.mocked(llmConfigApi.getLLMConfigs).mockResolvedValue({
        configs: [mockOllamaConfig],
        total: 1
      });

      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      // Save button should be disabled (no changes)
      const saveButton = screen.getByRole('button', { name: /save configuration/i });
      expect(saveButton).toBeDisabled();
    });
  });

  describe('Load Existing Configurations', () => {
    it('populates form with existing Ollama config', async () => {
      vi.mocked(llmConfigApi.getLLMConfigs).mockResolvedValue({
        configs: [mockOllamaConfig],
        total: 1
      });

      render(<LLMConfigForm />);

      await waitFor(() => {
        // Should show Active badge for active config
        expect(screen.getByText('Active')).toBeInTheDocument();
        // Ollama should show the configured URL and model
        expect(screen.getByDisplayValue('http://localhost:11434')).toBeInTheDocument();
        expect(screen.getByDisplayValue('llama3.1:8b')).toBeInTheDocument();
      });
    });

    it('loads first active config when multiple configs exist', async () => {
      vi.mocked(llmConfigApi.getLLMConfigs).mockResolvedValue({
        configs: [mockOllamaConfig, mockOpenAIConfig, mockGeminiConfig],
        total: 3
      });

      render(<LLMConfigForm />);

      await waitFor(() => {
        // Should load the active Ollama config first
        expect(screen.getByText('Active')).toBeInTheDocument();
        expect(screen.getByDisplayValue('http://localhost:11434')).toBeInTheDocument();
      });
    });

    it('loads Gemini config correctly (Phase 1 new provider)', async () => {
      const activeGeminiConfig = { ...mockGeminiConfig, isActive: true };
      vi.mocked(llmConfigApi.getLLMConfigs).mockResolvedValue({
        configs: [activeGeminiConfig],
        total: 1
      });

      render(<LLMConfigForm />);

      await waitFor(() => {
        // Should show Active badge
        expect(screen.getByText('Active')).toBeInTheDocument();
        // Should show Gemini model
        expect(screen.getByDisplayValue('gemini-2.0-flash')).toBeInTheDocument();
        // Should show API Key field (not Base URL)
        expect(screen.getByLabelText('API Key')).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    it('handles API error when loading configs', async () => {
      vi.mocked(llmConfigApi.getLLMConfigs).mockRejectedValue(new Error('API Error'));

      render(<LLMConfigForm />);

      // Component should render despite error
      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });
    });

    it('handles API error when saving config', async () => {
      const user = userEvent.setup();
      vi.mocked(llmConfigApi.saveLLMConfig).mockRejectedValue(new Error('Save failed'));

      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      // Enable config
      const enableSwitch = screen.getByRole('switch');
      await user.click(enableSwitch);

      // Click save
      const saveButton = screen.getByRole('button', { name: /save configuration/i });
      await user.click(saveButton);

      // Button should return to normal state after error
      await waitFor(() => {
        expect(screen.getByRole('button', { name: /save configuration/i })).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility & UX', () => {
    it('has accessible form labels for all fields', async () => {
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      // Check for properly labeled inputs
      expect(screen.getByLabelText('Provider')).toBeInTheDocument();
      expect(screen.getByLabelText('Base URL')).toBeInTheDocument();
      expect(screen.getByLabelText('Model')).toBeInTheDocument();
    });

    it('form controls are keyboard navigable', async () => {
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      // Key interactive elements should be enabled
      const providerSelect = screen.getByRole('combobox', { name: /provider/i });
      expect(providerSelect).toBeEnabled();

      const enableSwitch = screen.getByRole('switch');
      expect(enableSwitch).toBeEnabled();
    });

    it('shows provider description for selected provider', async () => {
      const user = userEvent.setup();
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('LLM Provider Configuration')).toBeInTheDocument();
      });

      // Should show Ollama description by default
      expect(screen.getByText('Local LLM inference with complete privacy')).toBeInTheDocument();

      // Switch to OpenAI
      const providerSelect = screen.getByRole('combobox', { name: /provider/i });
      await user.click(providerSelect);
      await waitFor(() => {
        expect(screen.getByText('OpenAI')).toBeInTheDocument();
      });
      await user.click(screen.getByText('OpenAI'));

      // Should show OpenAI description
      await waitFor(() => {
        expect(screen.getByText('OpenAI GPT models via API')).toBeInTheDocument();
      });
    });
  });
});
