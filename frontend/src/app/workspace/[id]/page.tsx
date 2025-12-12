/**
 * Workspace detail page
 * 
 * Displays a single workspace with document upload functionality.
 * Users can drag-and-drop or select files to upload to the workspace.
 */

'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { IngestionDropzone } from '@/components/IngestionDropzone';
import { UploadProgress, type Upload } from '@/components/UploadProgress';
import { getWorkspace } from '@/lib/api';
import { uploadDocuments } from '@/lib/documents';
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
  const [isUploading, setIsUploading] = useState(false);

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

  const handleUploadComplete = (documents: any[]) => {
    // Update uploads to show success
    setUploads(prev => 
      prev.map(upload => ({
        ...upload,
        progress: 100,
        status: 'success' as const,
      }))
    );

    toast.success(`Successfully uploaded ${documents.length} ${documents.length === 1 ? 'file' : 'files'}`);
    
    // Clear uploads after 3 seconds
    setTimeout(() => {
      setUploads([]);
      setIsUploading(false);
    }, 3000);
  };

  const handleUploadError = (error: Error) => {
    // Mark all uploads as error
    setUploads(prev =>
      prev.map(upload => ({
        ...upload,
        status: 'error' as const,
        error: error.message,
      }))
    );
    setIsUploading(false);
  };

  const handleCancel = (fileName: string) => {
    // Remove the upload from the list
    setUploads(prev => prev.filter(upload => upload.fileName !== fileName));
    toast.info(`Cancelled upload of ${fileName}`);
  };

  const handleRetry = async (fileName: string) => {
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
    setIsUploading(true);

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
      <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
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

              {/* Document list placeholder */}
              <div className="mt-8 rounded-lg border border-dashed border-zinc-300 p-8 text-center dark:border-zinc-700">
                <p className="text-sm text-zinc-600 dark:text-zinc-400">
                  Document list coming in Story 2.4...
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
