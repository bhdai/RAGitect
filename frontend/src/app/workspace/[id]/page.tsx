/**
 * Workspace detail page
 * 
 * Displays a single workspace with document upload, listing, viewing,
 * and delete functionality. Implements a side-by-side layout with
 * document viewer panel (FR9).
 */

'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { IngestionDropzone } from '@/components/IngestionDropzone';
import { UploadProgress, type Upload } from '@/components/UploadProgress';
import { DocumentList } from '@/components/DocumentList';
import { DocumentViewer } from '@/components/DocumentViewer';
import { DeleteDocumentDialog } from '@/components/DeleteDocumentDialog';
import { getWorkspace } from '@/lib/api';
import { uploadDocuments, getDocumentStatus, deleteDocument, type Document } from '@/lib/documents';
import { toast } from 'sonner';
import type { Workspace } from '@/lib/types';

export default function WorkspacePage() {
  // Next.js App Router: use useParams hook for client components
  const params = useParams<{ id: string }>();
  const id = params.id;
  const [workspace, setWorkspace] = useState<Workspace | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploads, setUploads] = useState<Upload[]>([]);
  // Use ref for polling intervals to avoid stale closure issues
  const pollingIntervalsRef = useRef<Map<string, NodeJS.Timeout>>(new Map());
  
  // Document viewing and deletion state
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);
  const [documentToDelete, setDocumentToDelete] = useState<Document | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [documentListRefresh, setDocumentListRefresh] = useState(0);

  useEffect(() => {
    async function fetchWorkspace() {
      setIsLoading(true);
      setError(null);
      
      try {
        const data = await getWorkspace(id);
        setWorkspace(data);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load workspace';
        setError(message);
      } finally {
        setIsLoading(false);
      }
    }

    fetchWorkspace();
  }, [id]);

  // Cleanup polling intervals on unmount
  useEffect(() => {
    const intervalsRef = pollingIntervalsRef;
    return () => {
      intervalsRef.current.forEach(interval => clearInterval(interval));
      intervalsRef.current.clear();
    };
  }, []);

  const pollDocumentStatus = useCallback(async (documentId: string, fileName: string) => {
    try {
      const status = await getDocumentStatus(documentId);
      
      // Update upload status based on processing state
      setUploads(prev =>
        prev.map(upload =>
          upload.fileName === fileName
            ? {
                ...upload,
                status: (status.status === 'processing' || status.status === 'embedding') ? 'uploading' : 
                        status.status === 'ready' ? 'success' :
                        status.status === 'error' ? 'error' : upload.status,
                progress: (status.status === 'processing' || status.status === 'embedding') ? 95 : 
                         status.status === 'ready' ? 100 : upload.progress,
                phase: status.phase ?? undefined, // Pass phase for detailed progress indicator
              }
            : upload
        )
      );

      // Stop polling if status is terminal
      if (status.status === 'ready' || status.status === 'error') {
        const interval = pollingIntervalsRef.current.get(documentId);
        if (interval) {
          clearInterval(interval);
          pollingIntervalsRef.current.delete(documentId);
        }

        if (status.status === 'ready') {
          toast.success(`Document ready: ${fileName}`);
        } else if (status.status === 'error') {
          toast.error(`Processing failed: ${fileName}`);
        }
      }
    } catch (err) {
      console.error(`Failed to poll status for ${documentId}:`, err);
    }
  }, []);

  const handleUploadComplete = useCallback((documents: Document[]) => {
    // Update uploads to show upload complete (100%)
    setUploads(prev => 
      prev.map(upload => ({
        ...upload,
        progress: 100,
        status: 'success' as const,
      }))
    );

    toast.success(`Uploaded ${documents.length} ${documents.length === 1 ? 'file' : 'files'} - parsing...`);
    
    // Refresh document list after upload
    setDocumentListRefresh(prev => prev + 1);
    
    // Start polling for each document
    documents.forEach(doc => {
      const interval = setInterval(() => {
        pollDocumentStatus(doc.id, doc.fileName);
      }, 2000); // Poll every 2 seconds

      pollingIntervalsRef.current.set(doc.id, interval);

      // Initial poll
      pollDocumentStatus(doc.id, doc.fileName);
    });
    
    // Don't auto-clear uploads anymore - wait for processing to complete
  }, [pollDocumentStatus]);

  const handleUploadError = (error: Error) => {
    // Mark all uploads as error
    setUploads(prev =>
      prev.map(upload => ({
        ...upload,
        status: 'error' as const,
        error: error.message,
      }))
    );
  };

  const handleCancel = (fileName: string) => {
    // Remove the upload from the list
    setUploads(prev => prev.filter(upload => upload.fileName !== fileName));
    toast.info(`Cancelled upload of ${fileName}`);
  };

  const handleRetry = async (_fileName: string) => { // eslint-disable-line @typescript-eslint/no-unused-vars
    // For now, just show a message - full retry logic would need file reference
    toast.info('Retry functionality will be available in the next iteration');
  };

  // Handle file selection from dropzone
  const handleFilesSelected = async (files: File[]) => {
    if (!workspace) return;

    // Initialize uploads
    const newUploads: Upload[] = files.map(file => ({
      fileName: file.name,
      progress: 0,
      status: 'uploading' as const,
      size: file.size,
    }));
    
    setUploads(newUploads);

    try {
      // Simulate progress (since we don't have real progress tracking yet)
      const progressInterval = setInterval(() => {
        setUploads(prev =>
          prev.map(upload => ({
            ...upload,
            progress: Math.min(upload.progress + 10, 90), // Cap at 90% until complete
          }))
        );
      }, 200);

      // Perform the upload
      const response = await uploadDocuments(workspace.id, files);
      
      clearInterval(progressInterval);
      handleUploadComplete(response.documents);
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Upload failed');
      handleUploadError(error);
      toast.error(error.message);
    }
  };

  // Handle document selection for viewing
  const handleSelectDocument = (doc: Document) => {
    setSelectedDocumentId(doc.id);
  };

  // Handle document deletion request (opens dialog)
  const handleDeleteDocument = (doc: Document) => {
    setDocumentToDelete(doc);
  };

  // Handle delete confirmation
  const handleDeleteConfirm = async () => {
    if (!documentToDelete) return;

    setIsDeleting(true);
    try {
      await deleteDocument(documentToDelete.id);
      toast.success(`Deleted ${documentToDelete.fileName}`);
      
      // Close viewer if deleted document was being viewed
      if (selectedDocumentId === documentToDelete.id) {
        setSelectedDocumentId(null);
      }
      
      // Refresh document list
      setDocumentListRefresh(prev => prev + 1);
      setDocumentToDelete(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete document';
      toast.error(message);
    } finally {
      setIsDeleting(false);
    }
  };

  // Handle delete cancellation
  const handleDeleteCancel = () => {
    setDocumentToDelete(null);
  };

  // Handle document viewer close
  const handleCloseViewer = () => {
    setSelectedDocumentId(null);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950">
        <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="mx-auto h-8 w-8 animate-spin rounded-full border-4 border-zinc-300 border-t-zinc-900 dark:border-zinc-600 dark:border-t-zinc-100" />
              <p className="mt-4 text-sm text-zinc-600 dark:text-zinc-400">
                Loading workspace...
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950">
        <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
          <div className="rounded-lg border border-red-200 bg-red-50 p-6 text-center dark:border-red-800 dark:bg-red-950">
            <p className="text-red-600 dark:text-red-400">{error}</p>
            <Link href="/">
              <Button variant="outline" className="mt-4">
                Back to Dashboard
              </Button>
            </Link>
          </div>
        </div>
      </div>
    );
  }

  if (!workspace) {
    return null;
  }

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950">
      {/* Main layout: flex container with optional document viewer panel */}
      <div className={`flex min-h-screen ${selectedDocumentId ? '' : ''}`}>
        {/* Main content area */}
        <div className="flex-1 overflow-auto">
          <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-8">
            {/* Header with back navigation */}
            <div className="mb-6">
              <Link 
                href="/"
                className="text-sm text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
              >
                ← Back to Dashboard
              </Link>
            </div>

            {/* Workspace info */}
            <Card>
              <CardHeader>
                <CardTitle className="text-2xl">{workspace.name}</CardTitle>
                {workspace.description && (
                  <CardDescription className="text-base">
                    {workspace.description}
                  </CardDescription>
                )}
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="text-sm text-muted-foreground">
                  <p>Created: {new Date(workspace.createdAt).toLocaleDateString()}</p>
                  <p>Last updated: {new Date(workspace.updatedAt).toLocaleDateString()}</p>
                </div>
                
                {/* Document upload section */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Documents</h3>
                  
                  {/* Upload progress */}
                  {uploads.length > 0 && (
                    <UploadProgress
                      uploads={uploads}
                      onCancel={handleCancel}
                      onRetry={handleRetry}
                    />
                  )}

                  {/* Dropzone */}
                  <IngestionDropzone
                    workspaceId={workspace.id}
                    onFilesSelected={handleFilesSelected}
                    onUploadComplete={handleUploadComplete}
                    onUploadError={handleUploadError}
                  />

                  {/* Document list */}
                  <div className="mt-6">
                    <DocumentList
                      workspaceId={workspace.id}
                      onSelectDocument={handleSelectDocument}
                      onDeleteDocument={handleDeleteDocument}
                      refreshTrigger={documentListRefresh}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Document viewer panel (conditionally rendered) */}
        {selectedDocumentId && (
          <div className="w-[700px] flex-shrink-0 border-l border-zinc-200 dark:border-zinc-800">
            <DocumentViewer
              documentId={selectedDocumentId}
              onClose={handleCloseViewer}
            />
          </div>
        )}
      </div>

      {/* Delete confirmation dialog */}
      <DeleteDocumentDialog
        document={documentToDelete}
        open={documentToDelete !== null}
        onConfirm={handleDeleteConfirm}
        onCancel={handleDeleteCancel}
        isDeleting={isDeleting}
      />
    </div>
  );
}
