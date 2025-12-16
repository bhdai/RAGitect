/**
 * UploadModal component
 * 
 * Modal dialog wrapping the IngestionDropzone for document uploads.
 * Triggered from the sidebar "Add Source" button per UX spec.
 */

'use client';

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { IngestionDropzone } from '@/components/IngestionDropzone';
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
}

/**
 * Modal dialog for document upload.
 * Wraps IngestionDropzone in a Dialog component.
 */
export function UploadModal({
  open,
  onOpenChange,
  workspaceId,
  onFilesSelected,
  onUploadComplete,
  onUploadError,
}: UploadModalProps) {
  const handleFilesSelected = (files: File[]) => {
    onFilesSelected(files);
    // Close modal after files are selected (upload will continue in background)
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent 
        className="sm:max-w-[500px]"
        data-testid="upload-modal"
      >
        <DialogHeader>
          <DialogTitle>Add Documents</DialogTitle>
          <DialogDescription>
            Upload documents to your workspace for RAG-powered conversations.
          </DialogDescription>
        </DialogHeader>
        
        <div className="mt-4">
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
