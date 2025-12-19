/**
 * Tests for ChatProviderSelector component
 *
 * Provider Configuration UX Improvements - Chat Provider Selector
 * These tests now use mocked context values
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ChatProviderSelector } from '../ChatProviderSelector';

// Mock the context module
const mockSelectProvider = vi.fn();
vi.mock('@/contexts/ProviderSelectionContext', () => ({
  useProviderSelectionContext: vi.fn(),
}));

import { useProviderSelectionContext } from '@/contexts/ProviderSelectionContext';

describe('ChatProviderSelector', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  it('shows loading state when isLoading is true', () => {
    vi.mocked(useProviderSelectionContext).mockReturnValue({
      selectedProvider: null,
      selectProvider: mockSelectProvider,
      providers: [],
      currentProvider: undefined,
      isLoading: true,
      error: null,
    });

    render(<ChatProviderSelector />);
    
    // Should show spinner
    const spinner = document.querySelector('.animate-spin');
    expect(spinner).toBeInTheDocument();
  });

  it('shows error state when loading fails', () => {
    vi.mocked(useProviderSelectionContext).mockReturnValue({
      selectedProvider: null,
      selectProvider: mockSelectProvider,
      providers: [],
      currentProvider: undefined,
      isLoading: false,
      error: new Error('Network error'),
    });

    render(<ChatProviderSelector />);
    
    // Should show error message
    expect(screen.getByText('Failed to load models')).toBeInTheDocument();
    
    // Should show error icon (AlertCircle)
    const errorContainer = screen.getByText('Failed to load models').parentElement;
    expect(errorContainer).toHaveClass('text-destructive');
  });

  it('shows "Configure model" link when no providers available', () => {
    vi.mocked(useProviderSelectionContext).mockReturnValue({
      selectedProvider: null,
      selectProvider: mockSelectProvider,
      providers: [],
      currentProvider: undefined,
      isLoading: false,
      error: null,
    });

    render(<ChatProviderSelector />);

    expect(screen.getByText('Configure model')).toBeInTheDocument();
    // Should link to settings
    const link = screen.getByRole('link');
    expect(link).toHaveAttribute('href', '/settings');
  });

  it('renders selector with active providers', () => {
    vi.mocked(useProviderSelectionContext).mockReturnValue({
      selectedProvider: 'openai',
      selectProvider: mockSelectProvider,
      providers: [
        { providerName: 'openai', displayName: 'OpenAI', model: 'gpt-4o', isActive: true },
        { providerName: 'anthropic', displayName: 'Anthropic', model: 'claude-sonnet-4-20250514', isActive: true },
      ],
      currentProvider: { providerName: 'openai', displayName: 'OpenAI', model: 'gpt-4o', isActive: true },
      isLoading: false,
      error: null,
    });

    render(<ChatProviderSelector />);

    // Should show the current provider's model
    expect(screen.getByText('gpt-4o')).toBeInTheDocument();
  });

  it('shows provider display name in dropdown options', async () => {
    const user = userEvent.setup();
    vi.mocked(useProviderSelectionContext).mockReturnValue({
      selectedProvider: 'openai',
      selectProvider: mockSelectProvider,
      providers: [
        { providerName: 'openai', displayName: 'OpenAI', model: 'gpt-4o', isActive: true },
        { providerName: 'anthropic', displayName: 'Anthropic', model: 'claude-sonnet-4-20250514', isActive: true },
      ],
      currentProvider: { providerName: 'openai', displayName: 'OpenAI', model: 'gpt-4o', isActive: true },
      isLoading: false,
      error: null,
    });

    render(<ChatProviderSelector />);

    // Open dropdown
    const trigger = screen.getByRole('combobox');
    await user.click(trigger);

    await waitFor(() => {
      expect(screen.getByText('(OpenAI)')).toBeInTheDocument();
      expect(screen.getByText('(Anthropic)')).toBeInTheDocument();
    });
  });

  it('calls selectProvider when option is selected', async () => {
    const user = userEvent.setup();
    vi.mocked(useProviderSelectionContext).mockReturnValue({
      selectedProvider: 'openai',
      selectProvider: mockSelectProvider,
      providers: [
        { providerName: 'openai', displayName: 'OpenAI', model: 'gpt-4o', isActive: true },
        { providerName: 'anthropic', displayName: 'Anthropic', model: 'claude-sonnet-4-20250514', isActive: true },
      ],
      currentProvider: { providerName: 'openai', displayName: 'OpenAI', model: 'gpt-4o', isActive: true },
      isLoading: false,
      error: null,
    });

    render(<ChatProviderSelector />);

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

    // Check selectProvider was called
    expect(mockSelectProvider).toHaveBeenCalledWith('anthropic');
  });

  it('shows "Select model" when no current provider', () => {
    vi.mocked(useProviderSelectionContext).mockReturnValue({
      selectedProvider: 'nonexistent',
      selectProvider: mockSelectProvider,
      providers: [
        { providerName: 'openai', displayName: 'OpenAI', model: 'gpt-4o', isActive: true },
      ],
      currentProvider: undefined, // No matching provider
      isLoading: false,
      error: null,
    });

    render(<ChatProviderSelector />);

    expect(screen.getByText('Select model')).toBeInTheDocument();
  });

  it('includes settings link in dropdown', async () => {
    const user = userEvent.setup();
    vi.mocked(useProviderSelectionContext).mockReturnValue({
      selectedProvider: 'openai',
      selectProvider: mockSelectProvider,
      providers: [
        { providerName: 'openai', displayName: 'OpenAI', model: 'gpt-4o', isActive: true },
      ],
      currentProvider: { providerName: 'openai', displayName: 'OpenAI', model: 'gpt-4o', isActive: true },
      isLoading: false,
      error: null,
    });

    render(<ChatProviderSelector />);

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
