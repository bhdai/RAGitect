/**
 * WorkspaceCard Unit Tests
 * 
 * Tests component rendering, delete functionality, and user interactions.
 * Following the testing strategy: Unit tests focus on component behavior,
 * while E2E tests verify full user flows.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { WorkspaceCard } from '../WorkspaceCard';
import type { Workspace } from '@/lib/types';

// Create mock router
const mockPush = vi.fn();
vi.mock('next/navigation', () => ({
  useRouter: vi.fn(() => ({
    push: mockPush,
  })),
}));

// Mock sonner toast
vi.mock('sonner', () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
  },
}));

// Reset mocks before each test
beforeEach(() => {
  vi.clearAllMocks();
});

const mockWorkspace: Workspace = {
  id: 'test-id-123',
  name: 'Test Workspace',
  description: 'A test workspace description',
  createdAt: '2025-12-01T10:00:00Z',
  updatedAt: '2025-12-01T10:00:00Z',
};

describe('WorkspaceCard', () => {
  it('renders workspace name and description', () => {
    render(<WorkspaceCard workspace={mockWorkspace} />);
    
    expect(screen.getByText('Test Workspace')).toBeInTheDocument();
    expect(screen.getByText('A test workspace description')).toBeInTheDocument();
  });

  it('renders formatted creation date', () => {
    render(<WorkspaceCard workspace={mockWorkspace} />);
    
    // Check for "Created" text followed by a formatted date
    expect(screen.getByText(/Created/i)).toBeInTheDocument();
    expect(screen.getByText(/Dec 1, 2025/i)).toBeInTheDocument();
  });

  it('calls onDelete handler when delete is confirmed', async () => {
    const user = userEvent.setup();
    const handleDelete = vi.fn().mockResolvedValue(undefined);
    render(<WorkspaceCard workspace={mockWorkspace} onDelete={handleDelete} />);
    
    // Open dropdown menu
    const menuButton = screen.getByLabelText(/More options for Test Workspace/i);
    await user.click(menuButton);
    
    // Click delete option
    const deleteMenuItem = await screen.findByText('Delete Workspace');
    await user.click(deleteMenuItem);
    
    // Dialog should now be open - find and click confirm button
    const confirmButton = await screen.findByRole('button', { name: /delete/i });
    await user.click(confirmButton);
    
    // Wait for delete handler to be called
    await waitFor(() => {
      expect(handleDelete).toHaveBeenCalledWith('test-id-123');
    });
  });

  it('does not delete when dialog is cancelled', async () => {
    const user = userEvent.setup();
    const handleDelete = vi.fn();
    render(<WorkspaceCard workspace={mockWorkspace} onDelete={handleDelete} />);
    
    // Open dropdown menu
    const menuButton = screen.getByLabelText(/More options for Test Workspace/i);
    await user.click(menuButton);
    
    // Click delete option
    const deleteMenuItem = await screen.findByText('Delete Workspace');
    await user.click(deleteMenuItem);
    
    // Find and click cancel button
    const cancelButton = await screen.findByRole('button', { name: /cancel/i });
    await user.click(cancelButton);
    
    // Delete handler should not be called
    expect(handleDelete).not.toHaveBeenCalled();
  });

  it('shows error toast when onDelete is not provided', async () => {
    const user = userEvent.setup();
    const { toast } = await import('sonner');
    render(<WorkspaceCard workspace={mockWorkspace} />);
    
    // Open dropdown and click delete
    const menuButton = screen.getByLabelText(/More options for Test Workspace/i);
    await user.click(menuButton);
    const deleteMenuItem = await screen.findByText('Delete Workspace');
    await user.click(deleteMenuItem);
    
    // Confirm deletion
    const confirmButton = await screen.findByRole('button', { name: /delete/i });
    await user.click(confirmButton);
    
    // Should show error toast
    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith('Delete function not available');
    });
  });

  it('handles keyboard navigation with Enter key', async () => {
    const user = userEvent.setup();
    render(<WorkspaceCard workspace={mockWorkspace} />);
    
    const card = screen.getByRole('button', { name: /Open workspace Test Workspace/i });
    await user.type(card, '{Enter}');
    
    expect(mockPush).toHaveBeenCalledWith('/workspace/test-id-123');
  });

  it('handles keyboard navigation with Space key', async () => {
    const user = userEvent.setup();
    render(<WorkspaceCard workspace={mockWorkspace} />);
    
    const card = screen.getByRole('button', { name: /Open workspace Test Workspace/i });
    // Focus the element first, then press space
    card.focus();
    await user.keyboard(' ');
    
    expect(mockPush).toHaveBeenCalledWith('/workspace/test-id-123');
  });
});
