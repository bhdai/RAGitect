/**
 * Tests for UploadModal component
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { UploadModal } from '../UploadModal';

// Mock the IngestionDropzone component
vi.mock('../IngestionDropzone', () => ({
  IngestionDropzone: ({ onFilesSelected }: { onFilesSelected: (files: File[]) => void }) => (
    <div data-testid="ingestion-dropzone" onClick={() => onFilesSelected([])}>
      Drag and drop files here
    </div>
  ),
}));

describe('UploadModal', () => {
  const defaultProps = {
    open: true,
    onOpenChange: vi.fn(),
    workspaceId: 'test-workspace-id',
    onFilesSelected: vi.fn(),
    onUploadComplete: vi.fn(),
    onUploadError: vi.fn(),
  };

  it('renders the modal when open is true', () => {
    render(<UploadModal {...defaultProps} />);
    
    expect(screen.getByTestId('upload-modal')).toBeInTheDocument();
    expect(screen.getByText('Add Documents')).toBeInTheDocument();
    expect(screen.getByText(/Upload documents to your workspace/)).toBeInTheDocument();
  });

  it('does not render modal content when open is false', () => {
    render(<UploadModal {...defaultProps} open={false} />);
    
    expect(screen.queryByTestId('upload-modal')).not.toBeInTheDocument();
  });

  it('renders the IngestionDropzone inside the modal', () => {
    render(<UploadModal {...defaultProps} />);
    
    expect(screen.getByTestId('ingestion-dropzone')).toBeInTheDocument();
    expect(screen.getByText('Drag and drop files here')).toBeInTheDocument();
  });

  it('calls onOpenChange with false when files are selected', () => {
    render(<UploadModal {...defaultProps} />);
    
    // Click the dropzone to simulate file selection
    fireEvent.click(screen.getByTestId('ingestion-dropzone'));
    
    expect(defaultProps.onFilesSelected).toHaveBeenCalled();
    expect(defaultProps.onOpenChange).toHaveBeenCalledWith(false);
  });

  it('has proper dialog title for accessibility', () => {
    render(<UploadModal {...defaultProps} />);
    
    expect(screen.getByRole('heading', { name: 'Add Documents' })).toBeInTheDocument();
  });
});
