/**
 * Tests for ChatProviderSelector component
 *
 * Provider Configuration UX Improvements - Chat Provider Selector
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ChatProviderSelector } from '../ChatProviderSelector';
import * as llmConfigApi from '@/lib/llmConfig';

// Mock the API module
vi.mock('@/lib/llmConfig', () => ({
  getLLMConfigs: vi.fn(),
}));

describe('ChatProviderSelector', () => {
  const mockConfigs: llmConfigApi.LLMProviderConfig[] = [
    {
      id: '1',
      providerName: 'openai',
      model: 'gpt-4o',
      isActive: true,
      createdAt: '2024-01-01T00:00:00Z',
      updatedAt: '2024-01-01T00:00:00Z',
    },
    {
      id: '2',
      providerName: 'anthropic',
      model: 'claude-sonnet-4-20250514',
      isActive: true,
      createdAt: '2024-01-01T00:00:00Z',
      updatedAt: '2024-01-01T00:00:00Z',
    },
    {
      id: '3',
      providerName: 'ollama',
      baseUrl: 'http://localhost:11434',
      model: 'llama3.1:8b',
      isActive: false, // Inactive, should not appear
      createdAt: '2024-01-01T00:00:00Z',
      updatedAt: '2024-01-01T00:00:00Z',
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    // Clear localStorage
    localStorage.clear();
  });

  it('shows loading state initially', () => {
    vi.mocked(llmConfigApi.getLLMConfigs).mockImplementation(
      () => new Promise(() => {}) // Never resolves to keep loading state
    );

    render(<ChatProviderSelector />);
    
    // Should show spinner
    const spinner = document.querySelector('.animate-spin');
    expect(spinner).toBeInTheDocument();
  });

  it('shows "Configure model" link when no providers configured', async () => {
    vi.mocked(llmConfigApi.getLLMConfigs).mockResolvedValue({
      configs: [],
      total: 0,
    });

    render(<ChatProviderSelector />);

    await waitFor(() => {
      expect(screen.getByText('Configure model')).toBeInTheDocument();
    });

    // Should link to settings
    const link = screen.getByRole('link');
    expect(link).toHaveAttribute('href', '/settings');
  });

  it('shows "Configure model" when all providers are inactive', async () => {
    vi.mocked(llmConfigApi.getLLMConfigs).mockResolvedValue({
      configs: [
        { ...mockConfigs[0], isActive: false },
        { ...mockConfigs[1], isActive: false },
      ],
      total: 2,
    });

    render(<ChatProviderSelector />);

    await waitFor(() => {
      expect(screen.getByText('Configure model')).toBeInTheDocument();
    });
  });

  it('renders selector with active providers', async () => {
    vi.mocked(llmConfigApi.getLLMConfigs).mockResolvedValue({
      configs: mockConfigs,
      total: 3,
    });

    render(<ChatProviderSelector />);

    await waitFor(() => {
      // Should show the first active provider's model
      expect(screen.getByText('gpt-4o')).toBeInTheDocument();
    });
  });

  it('only shows active providers in dropdown', async () => {
    const user = userEvent.setup();
    vi.mocked(llmConfigApi.getLLMConfigs).mockResolvedValue({
      configs: mockConfigs,
      total: 3,
    });

    render(<ChatProviderSelector />);

    await waitFor(() => {
      expect(screen.getByText('gpt-4o')).toBeInTheDocument();
    });

    // Open dropdown
    const trigger = screen.getByRole('combobox');
    await user.click(trigger);

    // Should show active providers (may have duplicates due to trigger + option)
    await waitFor(() => {
      expect(screen.getAllByText('gpt-4o').length).toBeGreaterThan(0);
      expect(screen.getAllByText('claude-sonnet-4-20250514').length).toBeGreaterThan(0);
    });

    // Inactive provider (ollama) should NOT be in dropdown
    expect(screen.queryByText('llama3.1:8b')).not.toBeInTheDocument();
  });

  it('shows provider display name in parentheses', async () => {
    const user = userEvent.setup();
    vi.mocked(llmConfigApi.getLLMConfigs).mockResolvedValue({
      configs: mockConfigs,
      total: 3,
    });

    render(<ChatProviderSelector />);

    await waitFor(() => {
      expect(screen.getByText('gpt-4o')).toBeInTheDocument();
    });

    // Open dropdown
    const trigger = screen.getByRole('combobox');
    await user.click(trigger);

    await waitFor(() => {
      expect(screen.getByText('(OpenAI)')).toBeInTheDocument();
      expect(screen.getByText('(Anthropic)')).toBeInTheDocument();
    });
  });

  it('persists selection to localStorage', async () => {
    const user = userEvent.setup();
    vi.mocked(llmConfigApi.getLLMConfigs).mockResolvedValue({
      configs: mockConfigs,
      total: 3,
    });

    render(<ChatProviderSelector />);

    await waitFor(() => {
      expect(screen.getByText('gpt-4o')).toBeInTheDocument();
    });

    // Open dropdown and select anthropic
    const trigger = screen.getByRole('combobox');
    await user.click(trigger);

    await waitFor(() => {
      expect(screen.getByText('claude-sonnet-4-20250514')).toBeInTheDocument();
    });

    // Find the SelectItem for Anthropic and click it
    const anthropicOption = screen.getAllByRole('option').find(
      opt => opt.textContent?.includes('claude-sonnet-4-20250514')
    );
    if (anthropicOption) {
      await user.click(anthropicOption);
    }

    // Check localStorage was updated
    await waitFor(() => {
      expect(localStorage.getItem('ragitect-selected-provider')).toBe('anthropic');
    });
  });

  it('loads saved selection from localStorage', async () => {
    // Pre-set localStorage
    localStorage.setItem('ragitect-selected-provider', 'anthropic');

    vi.mocked(llmConfigApi.getLLMConfigs).mockResolvedValue({
      configs: mockConfigs,
      total: 3,
    });

    render(<ChatProviderSelector />);

    await waitFor(() => {
      // Should show Anthropic's model since it was saved
      expect(screen.getByText('claude-sonnet-4-20250514')).toBeInTheDocument();
    });
  });

  it('falls back to first provider if saved provider no longer active', async () => {
    // Save a provider that is now inactive
    localStorage.setItem('ragitect-selected-provider', 'ollama');

    vi.mocked(llmConfigApi.getLLMConfigs).mockResolvedValue({
      configs: mockConfigs, // ollama is inactive in this list
      total: 3,
    });

    render(<ChatProviderSelector />);

    await waitFor(() => {
      // Should fall back to first active provider (openai)
      expect(screen.getByText('gpt-4o')).toBeInTheDocument();
    });
  });

  it('includes settings link in dropdown', async () => {
    const user = userEvent.setup();
    vi.mocked(llmConfigApi.getLLMConfigs).mockResolvedValue({
      configs: mockConfigs,
      total: 3,
    });

    render(<ChatProviderSelector />);

    await waitFor(() => {
      expect(screen.getByText('gpt-4o')).toBeInTheDocument();
    });

    // Open dropdown
    const trigger = screen.getByRole('combobox');
    await user.click(trigger);

    await waitFor(() => {
      expect(screen.getByText('Configure models...')).toBeInTheDocument();
    });

    // Should link to settings
    const settingsLink = screen.getByText('Configure models...').closest('a');
    expect(settingsLink).toHaveAttribute('href', '/settings');
  });
});
