/**
 * Dashboard page
 * 
 * Main page displaying all workspaces in a grid layout with
 * a "New Workspace" button to create new workspaces.
 */

'use client';

import { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { WorkspaceCard } from '@/components/WorkspaceCard';
import { CreateWorkspaceModal } from '@/components/CreateWorkspaceModal';
import { useWorkspaces } from '@/hooks/useWorkspaces';
import { useDeleteWorkspace } from '@/hooks/useWorkspaces';

export default function Dashboard() {
  const { workspaces, isLoading, error, refresh } = useWorkspaces();
  const { deleteWorkspace } = useDeleteWorkspace();
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleCreateSuccess = useCallback(() => {
    refresh();
  }, [refresh]);

  const handleDelete = useCallback(async (id: string) => {
    try {
      const success = await deleteWorkspace(id);
      if (success) {
        // Wait for refresh to complete before returning
        await refresh();
      }
    } catch (error) {
      // Error handling is done in the deleteWorkspace hook
      console.error('Delete failed:', error);
    }
  }, [deleteWorkspace, refresh]);

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950">
      <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50">
              RAGitect
            </h1>
            <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
              Manage your document workspaces
            </p>
          </div>
          <Button onClick={() => setIsModalOpen(true)}>
            New Workspace
          </Button>
        </div>

        {/* Content */}
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="mx-auto h-8 w-8 animate-spin rounded-full border-4 border-zinc-300 border-t-zinc-900 dark:border-zinc-600 dark:border-t-zinc-100" />
              <p className="mt-4 text-sm text-zinc-600 dark:text-zinc-400">
                Loading workspaces...
              </p>
            </div>
          </div>
        ) : error ? (
          <div className="rounded-lg border border-red-200 bg-red-50 p-6 text-center dark:border-red-800 dark:bg-red-950">
            <p className="text-red-600 dark:text-red-400">{error}</p>
            <Button 
              variant="outline" 
              className="mt-4"
              onClick={refresh}
            >
              Try Again
            </Button>
          </div>
        ) : workspaces.length === 0 ? (
          <div className="rounded-lg border border-dashed border-zinc-300 bg-zinc-100/50 p-12 text-center dark:border-zinc-700 dark:bg-zinc-900/50">
            <h3 className="text-lg font-medium text-zinc-900 dark:text-zinc-50">
              No workspaces yet
            </h3>
            <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
              Get started by creating your first workspace.
            </p>
            <Button 
              className="mt-4"
              onClick={() => setIsModalOpen(true)}
            >
              Create Workspace
            </Button>
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {workspaces.map((workspace) => (
              <WorkspaceCard 
                key={workspace.id} 
                workspace={workspace}
                onDelete={handleDelete}
              />
            ))}
          </div>
        )}
      </div>

      {/* Create Workspace Modal */}
      <CreateWorkspaceModal
        open={isModalOpen}
        onOpenChange={setIsModalOpen}
        onSuccess={handleCreateSuccess}
      />
    </div>
  );
}
