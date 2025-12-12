/**
 * IngestionDropzone component
 * 
 * Provides drag-and-drop file upload interface for document ingestion.
 * Supports multiple file formats: PDF, DOCX, TXT, MD, PPTX, XLSX, HTML.
 * Visual states: idle, drag-over, uploading, error.
 */

'use client';

import { useState, useRef, DragEvent, ChangeEvent } from 'react';
import { Upload, FileText, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';

type DropzoneState = 'idle' | 'drag-over' | 'uploading' | 'error';

interface IngestionDropzoneProps {
  workspaceId: string;
  onUploadComplete: (documents: any[]) => void;
  onUploadError: (error: Error) => void;
  onFilesSelected?: (files: File[]) => void; // Callback when files are validated
  maxFileSize?: number; // bytes, default 50MB
  acceptedTypes?: string[];
}

const DEFAULT_MAX_SIZE = 50 * 1024 * 1024; // 50MB
const DEFAULT_ACCEPTED_TYPES = [
  '.pdf',
  '.docx',
  '.txt',
  '.md',
  '.markdown',
  '.pptx',
  '.xlsx',
  '.html',
  '.htm',
];

const SUPPORTED_MIME_TYPES = [
  'application/pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'text/plain',
  'text/markdown',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  'text/html',
];

export function IngestionDropzone({
  workspaceId,
  onUploadComplete,
  onUploadError,
  onFilesSelected,
  maxFileSize = DEFAULT_MAX_SIZE,
  acceptedTypes = DEFAULT_ACCEPTED_TYPES,
}: IngestionDropzoneProps) {
  const [state, setState] = useState<DropzoneState>('idle');
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): { valid: boolean; error?: string } => {
    // Check file size
    if (file.size > maxFileSize) {
      return {
        valid: false,
        error: `File "${file.name}" exceeds maximum size of ${Math.round(maxFileSize / 1024 / 1024)}MB`,
      };
    }

    // Check file type by extension
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!acceptedTypes.includes(extension)) {
      return {
        valid: false,
        error: `File type "${extension}" is not supported. Supported types: ${acceptedTypes.join(', ')}`,
      };
    }
    return { valid: true };
  };

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;

    const fileArray = Array.from(files);
    const validFiles: File[] = [];
    const errors: string[] = [];

    // Validate each file
    for (const file of fileArray) {
      const validation = validateFile(file);
      if (validation.valid) {
        validFiles.push(file);
      } else {
        errors.push(validation.error!);
      }
    }

    // Show errors if any
    if (errors.length > 0) {
      errors.forEach((error) => toast.error(error));
      onUploadError(new Error(errors.join('; ')));
      setState('error');
      setTimeout(() => setState('idle'), 2000);
      return;
    }

    // Update selected files
    setSelectedFiles(validFiles);
    setState('idle');

    // Trigger upload callback if provided
    if (onFilesSelected && validFiles.length > 0) {
      onFilesSelected(validFiles);
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setState('drag-over');
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setState('idle');
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setState('idle');

    const files = e.dataTransfer.files;
    handleFiles(files);
  };

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    handleFiles(e.target.files);
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick();
    }
  };

  return (
    <div className="w-full">
      <div
        data-testid="dropzone"
        className={cn(
          'relative flex flex-col items-center justify-center w-full min-h-[200px] rounded-lg transition-all cursor-pointer',
          'focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
          state === 'idle' &&
            'border-2 border-dashed border-muted-foreground/25 hover:border-muted-foreground/50',
          state === 'drag-over' &&
            'border-2 border-solid border-primary bg-primary/5 scale-[1.02]',
          state === 'error' && 'border-2 border-solid border-destructive',
          state === 'uploading' && 'opacity-50 cursor-not-allowed'
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        tabIndex={0}
        role="button"
        aria-label="File upload dropzone"
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={acceptedTypes.join(',')}
          onChange={handleFileSelect}
          className="hidden"
          data-testid="file-input"
          disabled={state === 'uploading'}
        />

        <div className="flex flex-col items-center gap-2 p-8">
          {state === 'error' ? (
            <AlertCircle className="h-12 w-12 text-destructive" />
          ) : (
            <Upload className="h-12 w-12 text-muted-foreground" />
          )}

          <div className="text-center">
            <p className="text-sm font-medium">
              {state === 'drag-over'
                ? 'Drop files here'
                : state === 'error'
                  ? 'Error validating files'
                  : 'Drop files here'}
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              or click to select files
            </p>
          </div>

          <p className="text-xs text-muted-foreground mt-2">
            Supported: PDF, DOCX, TXT, MD, PPTX, XLSX, HTML
          </p>
          <p className="text-xs text-muted-foreground">
            Max size: {Math.round(maxFileSize / 1024 / 1024)}MB per file
          </p>
        </div>
      </div>

      {/* Display selected files */}
      {selectedFiles.length > 0 && (
        <div className="mt-4 space-y-2">
          <p className="text-sm font-medium">Selected files:</p>
          {selectedFiles.map((file, index) => (
            <div
              key={`${file.name}-${index}`}
              className="flex items-center gap-2 p-2 rounded-md bg-muted"
            >
              <FileText className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm flex-1">{file.name}</span>
              <span className="text-xs text-muted-foreground">
                {(file.size / 1024).toFixed(1)} KB
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
