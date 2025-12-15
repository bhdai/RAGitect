/**
 * Tests for DocumentSidebar component
 * 
 * Story 3.0: Streaming Infrastructure - AC3, AC4
 */

import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { DocumentSidebar } from '../DocumentSidebar';

// Mock the child components
vi.mock('../IngestionDropzone', () => ({
  IngestionDropzone: ({ onFilesSelected }: { onFilesSelected: (files: File[]) => void }) => (
    <div data-testid="ingestion-dropzone" onClick={() => onFilesSelected([])}>
      Drag and drop files here
    </div>
  ),
}));

vi.mock('../UploadProgress', () => ({
  UploadProgress: () => <div data-testid="upload-progress">Upload Progress</div>,
}));

vi.mock('../DocumentList', () => ({
  DocumentList: () => <div data-testid="document-list">Document List</div>,
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
    
    // Should have the sidebar container with proper width
    const sidebar = screen.getByTestId('document-sidebar');
    expect(sidebar).toBeInTheDocument();
    expect(sidebar).toHaveClass('w-72'); // 288px = 18rem
  });

  it('renders dropzone component', () => {
    render(<DocumentSidebar {...defaultProps} />);
    
    expect(screen.getByTestId('ingestion-dropzone')).toBeInTheDocument();
    expect(screen.getByText('Drag and drop files here')).toBeInTheDocument();
  });

  it('renders document list', () => {
    render(<DocumentSidebar {...defaultProps} />);
    
    expect(screen.getByTestId('document-list')).toBeInTheDocument();
  });

  it('renders upload progress when uploads exist', () => {
    const propsWithUploads = {
      ...defaultProps,
      uploads: [{ fileName: 'test.pdf', progress: 50, status: 'uploading' as const, size: 1000 }],
    };
    
    render(<DocumentSidebar {...propsWithUploads} />);
    
    expect(screen.getByTestId('upload-progress')).toBeInTheDocument();
  });

  it('does not render upload progress when no uploads', () => {
    render(<DocumentSidebar {...defaultProps} />);
    
    expect(screen.queryByTestId('upload-progress')).not.toBeInTheDocument();
  });

  it('renders Documents heading', () => {
    render(<DocumentSidebar {...defaultProps} />);
    
    expect(screen.getByRole('heading', { name: 'Documents' })).toBeInTheDocument();
  });

  it('has proper overflow handling for scrolling', () => {
    render(<DocumentSidebar {...defaultProps} />);
    
    const sidebar = screen.getByTestId('document-sidebar');
    expect(sidebar).toHaveClass('overflow-y-auto');
  });
});
