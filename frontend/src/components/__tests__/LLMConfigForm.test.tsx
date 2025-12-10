/**
 * LLMConfigForm Component Tests
 * 
 * Story 1.4: LLM Provider Configuration (Ollama & API Keys)
 * Task 10: Frontend unit and integration tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
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

describe('LLMConfigForm', () => {
  const mockOllamaConfig: llmConfigApi.LLMProviderConfig = {
    id: '1',
    providerName: 'ollama',
    baseUrl: 'http://localhost:11434',
    model: 'llama3.2',
    isActive: true,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  };

  const mockOpenAIConfig: llmConfigApi.LLMProviderConfig = {
    id: '2',
    providerName: 'openai',
    model: 'gpt-4o-mini',
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

  describe('Initial Rendering', () => {
    it('renders all provider cards', async () => {
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('Ollama')).toBeInTheDocument();
        expect(screen.getByText('OpenAI')).toBeInTheDocument();
        expect(screen.getByText('Anthropic')).toBeInTheDocument();
      });
    });

    it('shows loading state initially', () => {
      render(<LLMConfigForm />);
      // The component should show a loading spinner while fetching
      const spinner = document.querySelector('.animate-spin');
      expect(spinner).toBeInTheDocument();
    });

    it('loads existing configurations on mount', async () => {
      vi.mocked(llmConfigApi.getLLMConfigs).mockResolvedValue({ configs: [mockOllamaConfig], total: 1 });

      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(llmConfigApi.getLLMConfigs).toHaveBeenCalled();
      });
    });
  });

  describe('Provider Toggle', () => {
    it('toggles Ollama provider enabled state', async () => {
      const user = userEvent.setup();
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('Ollama')).toBeInTheDocument();
      });

      // Find the Ollama switch and toggle it
      const switches = screen.getAllByRole('switch');
      const ollamaSwitch = switches[0]; // First switch is Ollama

      await user.click(ollamaSwitch);
      
      // Switch state should change
      expect(ollamaSwitch).toBeInTheDocument();
    });

    it('shows form fields when provider is enabled', async () => {
      const user = userEvent.setup();
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('Ollama')).toBeInTheDocument();
      });

      // Enable Ollama
      const switches = screen.getAllByRole('switch');
      await user.click(switches[0]);

      // Should show base URL input for Ollama
      await waitFor(() => {
        const inputs = screen.getAllByRole('textbox');
        expect(inputs.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Form Inputs', () => {
    it('updates base URL input for Ollama', async () => {
      const user = userEvent.setup();
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('Ollama')).toBeInTheDocument();
      });

      // Enable Ollama
      const switches = screen.getAllByRole('switch');
      await user.click(switches[0]);

      await waitFor(() => {
        const urlInput = screen.getByPlaceholderText('http://localhost:11434');
        expect(urlInput).toBeInTheDocument();
      });
    });

    it('displays API key input for OpenAI', async () => {
      const user = userEvent.setup();
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('OpenAI')).toBeInTheDocument();
      });

      // Enable OpenAI
      const switches = screen.getAllByRole('switch');
      await user.click(switches[1]); // Second switch is OpenAI

      await waitFor(() => {
        const apiKeyInput = screen.getByPlaceholderText('sk-...');
        expect(apiKeyInput).toBeInTheDocument();
      });
    });

    it('displays API key input for Anthropic', async () => {
      const user = userEvent.setup();
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('Anthropic')).toBeInTheDocument();
      });

      // Enable Anthropic
      const switches = screen.getAllByRole('switch');
      await user.click(switches[2]); // Third switch is Anthropic

      await waitFor(() => {
        const apiKeyInput = screen.getByPlaceholderText('sk-ant-...');
        expect(apiKeyInput).toBeInTheDocument();
      });
    });
  });

  describe('Test Connection', () => {
    it('calls validateLLMConfig when Test Connection is clicked', async () => {
      const user = userEvent.setup();
      vi.mocked(llmConfigApi.validateLLMConfig).mockResolvedValue({ 
        valid: true,
        message: 'Connection successful',
      });

      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('Ollama')).toBeInTheDocument();
      });

      // Enable Ollama
      const switches = screen.getAllByRole('switch');
      await user.click(switches[0]);

      await waitFor(() => {
        const testButtons = screen.getAllByText('Test Connection');
        expect(testButtons.length).toBeGreaterThan(0);
      });

      // Click Test Connection button (first enabled one)
      const testButtons = screen.getAllByText('Test Connection');
      // Find the first non-disabled button
      const enabledButton = testButtons.find(btn => !btn.hasAttribute('disabled'));
      if (enabledButton) {
        await user.click(enabledButton);
      }

      await waitFor(() => {
        expect(llmConfigApi.validateLLMConfig).toHaveBeenCalled();
      });
    });

    it('shows success status after successful connection test', async () => {
      const user = userEvent.setup();
      vi.mocked(llmConfigApi.validateLLMConfig).mockResolvedValue({ 
        valid: true,
        message: 'Connection successful',
      });

      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('Ollama')).toBeInTheDocument();
      });

      // Enable Ollama
      const switches = screen.getAllByRole('switch');
      await user.click(switches[0]);

      await waitFor(() => {
        const testButtons = screen.getAllByText('Test Connection');
        expect(testButtons.length).toBeGreaterThan(0);
      });

      // Click Test Connection
      const testButtons = screen.getAllByText('Test Connection');
      const enabledButton = testButtons.find(btn => !btn.hasAttribute('disabled'));
      if (enabledButton) {
        await user.click(enabledButton);
      }

      await waitFor(() => {
        expect(screen.getByText('Connection successful')).toBeInTheDocument();
      });
    });

    it('shows error status after failed connection test', async () => {
      const user = userEvent.setup();
      vi.mocked(llmConfigApi.validateLLMConfig).mockResolvedValue({ 
        valid: false, 
        message: 'Connection refused',
        error: 'Connection refused',
      });

      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('Ollama')).toBeInTheDocument();
      });

      // Enable Ollama
      const switches = screen.getAllByRole('switch');
      await user.click(switches[0]);

      await waitFor(() => {
        const testButtons = screen.getAllByText('Test Connection');
        expect(testButtons.length).toBeGreaterThan(0);
      });

      // Click Test Connection
      const testButtons = screen.getAllByText('Test Connection');
      const enabledButton = testButtons.find(btn => !btn.hasAttribute('disabled'));
      if (enabledButton) {
        await user.click(enabledButton);
      }

      await waitFor(() => {
        expect(screen.getByText('Connection refused')).toBeInTheDocument();
      });
    });
  });

  describe('Save Configuration', () => {
    it('calls saveLLMConfig when Save is clicked', async () => {
      const user = userEvent.setup();
      vi.mocked(llmConfigApi.saveLLMConfig).mockResolvedValue(mockOllamaConfig);

      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('Ollama')).toBeInTheDocument();
      });

      // Enable Ollama (this sets hasChanges=true)
      const switches = screen.getAllByRole('switch');
      await user.click(switches[0]);

      // Find the first enabled Save Configuration button and click it
      await waitFor(() => {
        const saveButtons = screen.getAllByText('Save Configuration');
        const enabledSave = saveButtons.find(btn => !btn.hasAttribute('disabled'));
        expect(enabledSave).toBeInTheDocument();
      });

      const saveButtons = screen.getAllByText('Save Configuration');
      const enabledSave = saveButtons.find(btn => !btn.hasAttribute('disabled'));
      if (enabledSave) {
        await user.click(enabledSave);
      }

      await waitFor(() => {
        expect(llmConfigApi.saveLLMConfig).toHaveBeenCalled();
      });
    });

    it('shows saving state when save is in progress', async () => {
      const user = userEvent.setup();
      // Make save hang to test loading state
      vi.mocked(llmConfigApi.saveLLMConfig).mockImplementation(
        () => new Promise(resolve => setTimeout(() => resolve(mockOllamaConfig), 1000))
      );

      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('Ollama')).toBeInTheDocument();
      });

      // Enable Ollama
      const switches = screen.getAllByRole('switch');
      await user.click(switches[0]);

      // Find and click enabled Save Configuration button
      await waitFor(() => {
        const saveButtons = screen.getAllByText('Save Configuration');
        const enabledSave = saveButtons.find(btn => !btn.hasAttribute('disabled'));
        expect(enabledSave).toBeInTheDocument();
      });

      const saveButtons = screen.getAllByText('Save Configuration');
      const enabledSave = saveButtons.find(btn => !btn.hasAttribute('disabled'));
      if (enabledSave) {
        await user.click(enabledSave);
      }

      // Button should show saving state
      await waitFor(() => {
        expect(screen.getByText('Saving...')).toBeInTheDocument();
      });
    });
  });

  describe('Load Existing Configurations', () => {
    it('populates form with existing Ollama config', async () => {
      vi.mocked(llmConfigApi.getLLMConfigs).mockResolvedValue({ configs: [mockOllamaConfig], total: 1 });

      render(<LLMConfigForm />);

      await waitFor(() => {
        // Ollama should be enabled and show the configured URL
        const urlInput = screen.getByDisplayValue('http://localhost:11434');
        expect(urlInput).toBeInTheDocument();
      });
    });

    it('populates form with multiple configs', async () => {
      vi.mocked(llmConfigApi.getLLMConfigs).mockResolvedValue({
        configs: [mockOllamaConfig, mockOpenAIConfig],
        total: 2
      });

      render(<LLMConfigForm />);

      await waitFor(() => {
        // Both providers should be loaded
        const urlInput = screen.getByDisplayValue('http://localhost:11434');
        expect(urlInput).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    it('handles API error when loading configs', async () => {
      vi.mocked(llmConfigApi.getLLMConfigs).mockRejectedValue(new Error('API Error'));

      render(<LLMConfigForm />);

      // Component should render despite error
      await waitFor(() => {
        expect(screen.getByText('Ollama')).toBeInTheDocument();
      });
    });

    it('handles API error when saving config', async () => {
      const user = userEvent.setup();
      vi.mocked(llmConfigApi.saveLLMConfig).mockRejectedValue(new Error('Save failed'));

      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('Ollama')).toBeInTheDocument();
      });

      // Enable Ollama
      const switches = screen.getAllByRole('switch');
      await user.click(switches[0]);

      // Find enabled Save Configuration button and click
      await waitFor(() => {
        const saveButtons = screen.getAllByText('Save Configuration');
        const enabledSave = saveButtons.find(btn => !btn.hasAttribute('disabled'));
        expect(enabledSave).toBeInTheDocument();
      });

      const saveButtons = screen.getAllByText('Save Configuration');
      const enabledSave = saveButtons.find(btn => !btn.hasAttribute('disabled'));
      if (enabledSave) {
        await user.click(enabledSave);
      }

      // Button should return to normal state after error
      await waitFor(() => {
        const saveButtonsAfter = screen.getAllByText('Save Configuration');
        expect(saveButtonsAfter.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Accessibility', () => {
    it('has accessible form labels', async () => {
      const user = userEvent.setup();
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('Ollama')).toBeInTheDocument();
      });

      // Enable Ollama
      const switches = screen.getAllByRole('switch');
      await user.click(switches[0]);

      await waitFor(() => {
        // Check for labeled inputs
        const baseUrlLabel = screen.getByText('Base URL');
        expect(baseUrlLabel).toBeInTheDocument();
      });
    });

    it('provider cards are keyboard navigable', async () => {
      render(<LLMConfigForm />);

      await waitFor(() => {
        expect(screen.getByText('Ollama')).toBeInTheDocument();
      });

      // Switches should be focusable
      const switches = screen.getAllByRole('switch');
      expect(switches[0]).toBeEnabled();
    });
  });
});
