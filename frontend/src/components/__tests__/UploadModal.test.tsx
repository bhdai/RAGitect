/**
 * Tests for UploadModal component
 */

import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
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
    onUrlSubmit: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the modal when open is true', () => {
    render(<UploadModal {...defaultProps} />);
    
    expect(screen.getByTestId('upload-modal')).toBeInTheDocument();
    expect(screen.getByText('Add Documents')).toBeInTheDocument();
    expect(screen.getByText(/Upload documents or add URLs/)).toBeInTheDocument();
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

  // AC1: URL Input Field
  it('renders URL input field with correct label and placeholder', () => {
    render(<UploadModal {...defaultProps} />);

    const urlInput = screen.getByTestId('url-input');
    expect(urlInput).toBeInTheDocument();
    expect(urlInput).toHaveAttribute('placeholder', 'https://example.com/article');
    expect(screen.getByText('Enter URL (web page, YouTube, PDF)')).toBeInTheDocument();
  });

  it('renders URL input above the file dropzone', () => {
    render(<UploadModal {...defaultProps} />);

    // URL input should come before the dropzone in the DOM
    const modal = screen.getByTestId('upload-modal');
    const allElements = modal.querySelectorAll('[data-testid]');
    const urlIndex = Array.from(allElements).findIndex(el => el.getAttribute('data-testid') === 'url-input');
    const dropzoneIndex = Array.from(allElements).findIndex(el => el.getAttribute('data-testid') === 'ingestion-dropzone');
    expect(urlIndex).toBeGreaterThanOrEqual(0);
    expect(dropzoneIndex).toBeGreaterThanOrEqual(0);
    expect(urlIndex).toBeLessThan(dropzoneIndex);
  });

  // AC2: URL Type Detection Visual Badge
  it('shows Globe icon for regular web URLs', async () => {
    const user = userEvent.setup();
    render(<UploadModal {...defaultProps} />);

    const urlInput = screen.getByTestId('url-input');
    await user.type(urlInput, 'https://example.com/article');

    expect(screen.getByTestId('url-type-icon-web')).toBeInTheDocument();
  });

  it('shows YouTube icon for YouTube URLs', async () => {
    const user = userEvent.setup();
    render(<UploadModal {...defaultProps} />);

    const urlInput = screen.getByTestId('url-input');
    await user.type(urlInput, 'https://youtube.com/watch?v=abc');

    expect(screen.getByTestId('url-type-icon-youtube')).toBeInTheDocument();
  });

  it('shows PDF icon for PDF URLs', async () => {
    const user = userEvent.setup();
    render(<UploadModal {...defaultProps} />);

    const urlInput = screen.getByTestId('url-input');
    await user.type(urlInput, 'https://example.com/doc.pdf');

    expect(screen.getByTestId('url-type-icon-pdf')).toBeInTheDocument();
  });

  // AC3: Submit URL Button
  it('renders Add URL button', () => {
    render(<UploadModal {...defaultProps} />);

    expect(screen.getByTestId('add-url-button')).toBeInTheDocument();
    expect(screen.getByTestId('add-url-button')).toHaveTextContent('Add URL');
  });

  it('calls onUrlSubmit with url and sourceType when Add URL clicked', async () => {
    const user = userEvent.setup();
    render(<UploadModal {...defaultProps} />);

    const urlInput = screen.getByTestId('url-input');
    await user.type(urlInput, 'https://example.com/article');
    await user.click(screen.getByTestId('add-url-button'));

    expect(defaultProps.onUrlSubmit).toHaveBeenCalledWith('https://example.com/article', 'url');
  });

  it('clears URL input after successful submit', async () => {
    const user = userEvent.setup();
    render(<UploadModal {...defaultProps} />);

    const urlInput = screen.getByTestId('url-input');
    await user.type(urlInput, 'https://example.com/article');
    await user.click(screen.getByTestId('add-url-button'));

    expect(urlInput).toHaveValue('');
  });

  it('does not call onUrlSubmit when URL input is empty', async () => {
    const user = userEvent.setup();
    render(<UploadModal {...defaultProps} />);

    await user.click(screen.getByTestId('add-url-button'));

    expect(defaultProps.onUrlSubmit).not.toHaveBeenCalled();
  });

  // Visual separator
  it('renders visual separator between URL input and dropzone', () => {
    render(<UploadModal {...defaultProps} />);

    expect(screen.getByTestId('url-file-separator')).toBeInTheDocument();
    expect(screen.getByText('or')).toBeInTheDocument();
  });
});
