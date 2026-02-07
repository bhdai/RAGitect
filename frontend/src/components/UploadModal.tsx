/**
 * UploadModal component
 * 
 * Modal dialog wrapping the IngestionDropzone for document uploads.
 * Also includes URL input for web page, YouTube, and PDF ingestion.
 * Triggered from the sidebar "Add Source" button per UX spec.
 */

'use client';

import { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { IngestionDropzone } from '@/components/IngestionDropzone';
import { Globe, Play, FileType } from 'lucide-react';
import { detectUrlType } from '@/lib/documents';
import type { Document } from '@/lib/documents';

export interface UploadModalProps {
  /** Whether the modal is open */
  open: boolean;
  /** Callback to close the modal */
  onOpenChange: (open: boolean) => void;
  /** Workspace ID for document uploads */
  workspaceId: string;
  /** Callback when files are selected for upload */
  onFilesSelected: (files: File[]) => void;
  /** Callback when upload completes successfully */
  onUploadComplete: (documents: Document[]) => void;
  /** Callback when upload fails */
  onUploadError: (error: Error) => void;
  /** Callback when URL is submitted for ingestion */
  onUrlSubmit?: (url: string, sourceType: 'url' | 'youtube' | 'pdf') => void;
}

/**
 * Get the URL type icon component based on detected type
 */
function UrlTypeIcon({ urlType }: { urlType: 'youtube' | 'pdf' | 'url' }) {
  switch (urlType) {
    case 'youtube':
      return <Play className="h-4 w-4 text-red-500" data-testid="url-type-icon-youtube" />;
    case 'pdf':
      return <FileType className="h-4 w-4 text-orange-500" data-testid="url-type-icon-pdf" />;
    default:
      return <Globe className="h-4 w-4 text-blue-500" data-testid="url-type-icon-web" />;
  }
}

/**
 * Modal dialog for document upload.
 * Wraps IngestionDropzone in a Dialog component with URL input above.
 */
export function UploadModal({
  open,
  onOpenChange,
  workspaceId,
  onFilesSelected,
  onUploadComplete,
  onUploadError,
  onUrlSubmit,
}: UploadModalProps) {
  const [urlValue, setUrlValue] = useState('');

  const handleFilesSelected = (files: File[]) => {
    onFilesSelected(files);
    // Close modal after files are selected (upload will continue in background)
    onOpenChange(false);
  };

  const handleUrlSubmit = () => {
    const trimmed = urlValue.trim();
    if (!trimmed || !onUrlSubmit) return;
    const sourceType = detectUrlType(trimmed);
    onUrlSubmit(trimmed, sourceType);
    setUrlValue('');
  };

  const detectedType = urlValue.trim() ? detectUrlType(urlValue) : null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent 
        className="sm:max-w-[500px]"
        data-testid="upload-modal"
      >
        <DialogHeader>
          <DialogTitle>Add Documents</DialogTitle>
          <DialogDescription>
            Upload documents or add URLs to your workspace for RAG-powered conversations.
          </DialogDescription>
        </DialogHeader>
        
        <div className="mt-4 space-y-4">
          {/* URL Input Section */}
          <div className="space-y-2">
            <Label htmlFor="url-input">Enter URL (web page, YouTube, PDF)</Label>
            <div className="flex items-center gap-2">
              {detectedType && <UrlTypeIcon urlType={detectedType} />}
              <Input
                id="url-input"
                data-testid="url-input"
                type="text"
                placeholder="https://example.com/article"
                value={urlValue}
                onChange={(e) => setUrlValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleUrlSubmit();
                  }
                }}
              />
            </div>
            <Button
              data-testid="add-url-button"
              variant="default"
              size="sm"
              className="w-full"
              onClick={handleUrlSubmit}
              disabled={!urlValue.trim()}
            >
              Add URL
            </Button>
          </div>

          {/* Visual separator */}
          <div className="flex items-center gap-4" data-testid="url-file-separator">
            <div className="flex-1 h-px bg-border" />
            <span className="text-xs text-muted-foreground">or</span>
            <div className="flex-1 h-px bg-border" />
          </div>

          {/* File Dropzone */}
          <IngestionDropzone
            workspaceId={workspaceId}
            onFilesSelected={handleFilesSelected}
            onUploadComplete={onUploadComplete}
            onUploadError={onUploadError}
          />
        </div>
      </DialogContent>
    </Dialog>
  );
}
