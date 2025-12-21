/**
 * Tests for DocumentSidebar component
 *
 * Updated for collapsible sidebar with modal upload
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { DocumentSidebar } from '../DocumentSidebar';

// Mock the child components
vi.mock('../UploadModal', () => ({
  UploadModal: ({ open }: { open: boolean }) => (
    open ? <div data-testid="upload-modal">Upload Modal</div> : null
  ),
}));

vi.mock('../DocumentList', () => ({
  DocumentList: ({ collapsed }: { collapsed?: boolean }) => (
    <div data-testid="document-list" data-collapsed={collapsed}>
      Document List {collapsed ? '(collapsed)' : '(expanded)'}
    </div>
  ),
}));

// Mock useLocalStorage with a simple implementation
let mockIsCollapsed = false;
vi.mock('@/hooks/useLocalStorage', () => ({
  useLocalStorage: () => [mockIsCollapsed, (val: boolean) => { mockIsCollapsed = val; }],
}));

describe('DocumentSidebar', () => {
  const defaultProps = {
    workspaceId: 'test-workspace-id',
    onSelectDocument: vi.fn(),
    onDeleteDocument: vi.fn(),
    refreshTrigger: 0,
    uploads: [],
    onFilesSelected: vi.fn(),
    onUploadComplete: vi.fn(),
    onUploadError: vi.fn(),
    onCancelUpload: vi.fn(),
    onRetryUpload: vi.fn(),
  };

  it('renders the sidebar with correct structure', () => {
    render(<DocumentSidebar {...defaultProps} />);

    // Should have the sidebar container
    const sidebar = screen.getByTestId('document-sidebar');
    expect(sidebar).toBeInTheDocument();
  });

  it('renders the collapse toggle button', () => {
    render(<DocumentSidebar {...defaultProps} />);

    expect(screen.getByTestId('sidebar-toggle')).toBeInTheDocument();
  });

  it('renders Add Source button', () => {
    render(<DocumentSidebar {...defaultProps} />);

    expect(screen.getByTestId('add-source-button')).toBeInTheDocument();
    expect(screen.getByText('Add Source')).toBeInTheDocument();
  });

  it('opens upload modal when Add Source is clicked', () => {
    render(<DocumentSidebar {...defaultProps} />);

    // Modal should not be visible initially
    expect(screen.queryByTestId('upload-modal')).not.toBeInTheDocument();

    // Click Add Source
    fireEvent.click(screen.getByTestId('add-source-button'));

    // Modal should now be visible
    expect(screen.getByTestId('upload-modal')).toBeInTheDocument();
  });

  it('renders document list', () => {
    render(<DocumentSidebar {...defaultProps} />);

    expect(screen.getByTestId('document-list')).toBeInTheDocument();
  });

  it('accepts uploads prop for interface compatibility', () => {
    const propsWithUploads = {
      ...defaultProps,
      uploads: [{ fileName: 'test.pdf', progress: 50, status: 'uploading' as const, size: 1000 }],
    };

    // Should render without errors even with uploads (no longer displays UploadProgress)
    render(<DocumentSidebar {...propsWithUploads} />);

    expect(screen.getByTestId('document-sidebar')).toBeInTheDocument();
  });

  it('does not render upload progress component', () => {
    render(<DocumentSidebar {...defaultProps} />);

    // UploadProgress has been removed - status is now shown via DocumentList color coding
    expect(screen.queryByTestId('upload-progress')).not.toBeInTheDocument();
  });

  it('renders Documents heading', () => {
    render(<DocumentSidebar {...defaultProps} />);

    expect(screen.getByRole('heading', { name: 'Documents' })).toBeInTheDocument();
  });

  it('has proper structure for content layout', () => {
    render(<DocumentSidebar {...defaultProps} />);

    const sidebar = screen.getByTestId('document-sidebar');
    expect(sidebar).toHaveClass('flex-shrink-0');
    expect(sidebar).toHaveClass('h-full');
  });
});
