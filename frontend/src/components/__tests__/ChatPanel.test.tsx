/**
 * Tests for ChatPanel component
 * 
 * Story 3.0: Streaming Infrastructure - AC3
 */

import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ChatPanel } from '../ChatPanel';

describe('ChatPanel', () => {
  it('renders the chat panel placeholder', () => {
    render(<ChatPanel workspaceId="test-workspace" />);
    
    expect(screen.getByText('Chat with your documents')).toBeInTheDocument();
  });

  it('shows coming soon message', () => {
    render(<ChatPanel workspaceId="test-workspace" />);
    
    expect(screen.getByText('Coming in Story 3.1')).toBeInTheDocument();
  });

  it('has proper flex styling for centering', () => {
    render(<ChatPanel workspaceId="test-workspace" />);
    
    const panel = screen.getByTestId('chat-panel');
    expect(panel).toHaveClass('flex-1');
    expect(panel).toHaveClass('flex');
    expect(panel).toHaveClass('items-center');
    expect(panel).toHaveClass('justify-center');
  });

  it('renders message icon', () => {
    render(<ChatPanel workspaceId="test-workspace" />);
    
    // Icon should be rendered (lucide-react MessageSquare)
    const icon = document.querySelector('svg');
    expect(icon).toBeInTheDocument();
  });

  it('has proper overflow handling', () => {
    render(<ChatPanel workspaceId="test-workspace" />);
    
    const panel = screen.getByTestId('chat-panel');
    expect(panel).toHaveClass('overflow-y-auto');
  });
});
