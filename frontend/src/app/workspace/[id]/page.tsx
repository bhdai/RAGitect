/**
 * Workspace detail page
 * 
 * Displays a single workspace with its name and description.
 * This is a placeholder that will be expanded in future stories.
 */

'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { getWorkspace } from '@/lib/api';
import type { Workspace } from '@/lib/types';

export default function WorkspacePage() {
  // Next.js App Router: use useParams hook for client components
  const params = useParams<{ id: string }>();
  const id = params.id;
  const [workspace, setWorkspace] = useState<Workspace | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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
          <CardContent>
            <div className="text-sm text-muted-foreground">
              <p>Created: {new Date(workspace.createdAt).toLocaleDateString()}</p>
              <p>Last updated: {new Date(workspace.updatedAt).toLocaleDateString()}</p>
            </div>
            
            {/* Placeholder for future document functionality */}
            <div className="mt-8 rounded-lg border border-dashed border-zinc-300 p-8 text-center dark:border-zinc-700">
              <p className="text-sm text-zinc-600 dark:text-zinc-400">
                Document upload and management coming soon...
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
